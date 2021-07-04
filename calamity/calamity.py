import numpy as np
import tensorflow as tf
from pyuvdata import UVData, UVCal, UVFlag
from . import utils
import copy
import argparse
import itertools
import datetime
from pyuvdata import utils as uvutils
from .utils import echo
from .utils import PBARS
from . import cal_utils
from . import modeling
import multiprocessing


OPTIMIZERS = {
    "Adadelta": tf.optimizers.Adadelta,
    "Adam": tf.optimizers.Adam,
    "Adamax": tf.optimizers.Adamax,
    "Ftrl": tf.optimizers.Ftrl,
    "Nadam": tf.optimizers.Nadam,
    "SGD": tf.optimizers.SGD,
    "RMSprop": tf.optimizers.RMSprop,
}


def tensorize_fg_model_comps(
    fg_model_comps, ants_map, nfreqs, sparse_threshold=1e-1, dtype=np.float32, notebook_progressbar=False, verbose=False
):
    """Convert per-baseline model components into a Ndata x Ncomponent tensor

    Parameters
    ----------
    fg_model_comps: dictionary
        dictionary with keys that are tuples of tuples of 2-tuples (thats right, 3 levels)
        in the first level, each tuple represents a 'modeling group' visibilities in each
        modeling group are represented by a set of basis vectors that span all baselines in that
        group with elements raveled by baseline and then frequency. Each tuple in the modeling group is a
        'redundant group' representing visibilities that we will represent with identical component coefficients
        each element of each 'redundant group' is a 2-tuple antenna pair. Our formalism easily accomodates modeling
        visibilities as redundant or non redundant (one simply needs to make each redundant group length 1).
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    nfreqs: int, optional
        number of frequency channels
    sparse_threshold: float, optional
        if the number of non-zero elements / total number of elements is greater
        then this value, then use a dense representation rather then a sparse representation.
    dtype: numpy.dtype
        data-type to store in sparse tensor.
        default is np.float32

    Returns
    -------
    fg_model_mat:
        If number of nonzero elements is above sparse threshold: tf.sparse.SparseTensor
        sparse tensor object holding foreground modeling vectors with a dense shape of
        Nants^2 x Nfreqs x Nfg_comps

        If number of nonzero elements is not above sparse threshold: tf.Tensor
        object holding foreground modeling vectors with shape of
        Nants x Nants x Nfreqs x Nfg_comps
    """
    echo(
        f"{datetime.datetime.now()} Computing sparse foreground components matrix...\n",
        verbose=verbose,
    )

    sparse_number_of_elements = 0
    nvectors = 0
    nants_data = len(ants_map)
    echo("Determining number of non-zero elements", verbose=verbose)
    for modeling_grp in PBARS[notebook_progressbar](fg_model_comps):
        for vind in range(fg_model_comps[modeling_grp].shape[1]):
            for grpnum, red_grp in enumerate(modeling_grp):
                for f in range(nfreqs):
                    sparse_number_of_elements += 1
            nvectors += 1

    dense_shape = (int(nants_data ** 2.0 * nfreqs), nvectors)
    dense_number_of_elements = dense_shape[0] * dense_shape[1]

    sparseness = sparse_number_of_elements / dense_number_of_elements
    echo(f"Fraction of modeling matrix with nonzero values is {(sparseness):.4e}", verbose=verbose)
    echo(f"Generating map between i,j indices and foreground modeling keys", verbose=verbose)
    modeling_grps = {}
    red_grp_nums = {}
    start_inds = {}
    stop_inds = {}
    start_ind = 0
    for modeling_grp in fg_model_comps:
        stop_ind = start_ind + fg_model_comps[modeling_grp].shape[1]
        for red_grp_num, red_grp in enumerate(modeling_grp):
            for ap in red_grp:
                i, j = ants_map[ap[0]], ants_map[ap[1]]
                modeling_grps[(i, j)] = modeling_grp
                red_grp_nums[(i, j)] = red_grp_num
                start_inds[(i, j)] = start_ind
                stop_inds[(i, j)] = stop_ind
        start_ind = stop_ind
    ordered_ijs = sorted(list(modeling_grps.keys()))
    use_sparse = sparseness <= sparse_threshold
    if use_sparse:
        echo("Using Sparse Representation.")
        comp_inds = np.zeros((sparse_number_of_elements, 2), dtype=np.int32)
        comp_vals = np.zeros(sparse_number_of_elements, dtype=dtype)
    else:
        echo("Using Dense Representation.", verbose=verbose)
        fg_model_mat = np.zeros(dense_shape, dtype=dtype)
    spinds = None
    echo(f"{datetime.datetime.now()} Filling out modeling vectors...\n", verbose=verbose)
    for i, j in PBARS[notebook_progressbar](ordered_ijs):
        blind = i * nants_data + j
        grpnum = red_grp_nums[(i, j)]
        fitgrp = modeling_grps[(i, j)]
        start_ind = start_inds[(i, j)]
        stop_ind = stop_inds[(i, j)]
        nvecs = stop_ind - start_ind
        dinds = np.hstack([np.ones(nvecs) * dind for dind in np.arange(blind * nfreqs, (blind + 1) * nfreqs)]).astype(
            np.int32
        )
        matcols = np.hstack([np.arange(start_ind, stop_ind) for i in np.arange(nfreqs)]).astype(np.int32)
        bl_mvec = fg_model_comps[fitgrp][grpnum * nfreqs : (grpnum + 1) * nfreqs].astype(dtype).flatten()
        ninds = len(dinds)
        if use_sparse:
            if spinds is None:
                spinds = np.arange(ninds).astype(np.int32)
            else:
                spinds = np.arange(ninds).astype(np.int32) + spinds[-1] + 1
            comp_vals[spinds] = bl_mvec
            comp_inds[spinds, 0], comp_inds[spinds, 1] = dinds, matcols
        else:
            fg_model_mat[dinds, matcols] = bl_mvec

    if use_sparse:
        fg_model_mat = tf.sparse.SparseTensor(indices=comp_inds, values=comp_vals, dense_shape=dense_shape)
    else:
        # convert to 4-tensor if not using sparse representation.
        fg_model_mat = tf.convert_to_tensor(fg_model_mat.reshape(nants_data, nants_data, nfreqs, dense_shape[1]), dtype=dtype)
    return fg_model_mat


def tensorize_data(
    uvdata,
    red_grps,
    ants_map,
    polarization,
    time_index,
    data_scale_factor=1.0,
    weights=None,
    dtype=np.float32,
):
    """Convert data in uvdata object to a tensor

    Parameters
    ----------
    uvdata: UVData object
        UVData object containing data, flags, and nsamples to tensorize.
    red_grps: list of lists of int 2-tuples
        a list of lists of 2-tuples where all antenna pairs within each sublist
        are redundant with eachother. Assumes that conjugates are correctly taken.
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    polarization: str
        pol-str of gain to extract.
    data_scale_factor: float, optional
        overall scaling factor to divide tensorized data by.
        default is 1.0
    weights: UVFlag object, optional
        UVFlag object wgts array contains weights for fitting.
        default is None -> weight by nsamples x ~flags
    dtype: numpy.dtype
        data-type to store in sparse tensor.
        default is np.float32

    Returns
    -------
    data_r: tf.Tensor object
        tf.Tensor object storing real parts of visibilities with shape Nants_data x Nants_data x Nfreqs
        scaled by data_scale_factor
    data_i: tf.Tensor object
        tf.Tensor object storing imag parts of visibilities with shape Nants_data x Nants_data x Nfreqs
        scaled by data_scale_factor
    wgts: tf.Tensor object
        tf.Tensor object storing wgts with shape Nants_data x Nants_data x Nfreqs
    """
    dshape = (uvdata.Nants_data, uvdata.Nants_data, uvdata.Nfreqs)
    data_r = np.zeros(dshape, dtype=dtype)
    data_i = np.zeros_like(data_r)
    wgts = np.zeros_like(data_r)
    wgtsum = 0.0
    for red_grp in red_grps:
        for ap in red_grp:
            bl = ap + (polarization,)
            data = uvdata.get_data(bl)[time_index] / data_scale_factor
            iflags = (~uvdata.get_flags(bl))[time_index].astype(dtype)
            nsamples = uvdata.get_nsamples(bl)[time_index].astype(dtype)
            i, j = ants_map[ap[0]], ants_map[ap[1]]
            data_r[i, j] = data.real.astype(dtype)
            data_i[i, j] = data.imag.astype(dtype)
            if weights is None:
                wgts[i, j] = iflags * nsamples
            else:
                if ap in weights.get_antpairs():
                    dinds = weights.antpair2ind(*ap)
                else:
                    dinds = weights.antpair2ind(*ap[::-1])
                polnum = np.where(
                    weights.polarization_array == uvutils.polstr2num(polarization, x_orientation=weights.x_orientation)
                )[0][0]
                wgts[i, j] = weights.weights_array[dinds[time_index], 0, :, polnum].astype(dtype) * iflags
            wgtsum += np.sum(wgts[i, j])
    data_r = tf.convert_to_tensor(data_r, dtype=dtype)
    data_i = tf.convert_to_tensor(data_i, dtype=dtype)
    wgts = tf.convert_to_tensor(wgts / wgtsum, dtype=dtype)
    return data_r, data_i, wgts


def renormalize(uvdata_reference_model, uvdata_deconv, gains, polarization, uvdata_flags=None):
    """Remove arbitrary phase and amplitude from deconvolved model and gains.

    Parameters
    ----------
    uvdata_reference_model: UVData object
        Reference model for "true" visibilities.
    uvdata_deconv: UVData object
        "Deconvolved" data solved for in self-cal loop.
    gains: UVCal object
        Gains solved for in self-cal loop.
    polarization: str
        Polarization string to compute phase and amplitude correction for.
    uvdata_flags: optional
        UVData object storing flags.
        Default is None -> use flags from uvdata_reference_model.

    Returns
    -------
    N/A: Modifies uvdata_deconv and gains in-place.
    """
    # compute and multiply out scale-factor accounting for overall amplitude and phase degeneracy.
    polnum_data = np.where(
        uvdata_deconv.polarization_array == uvutils.polstr2num(polarization, x_orientation=uvdata_deconv.x_orientation)
    )[0][0]

    if uvdata_flags is None:
        uvdata_flags = uvdata_reference_model

    selection = ~uvdata_flags.flag_array[:, :, :, polnum_data]

    scale_factor_phase = np.angle(
        np.mean(
            uvdata_reference_model.data_array[:, :, :, polnum_data][selection]
            / uvdata_deconv.data_array[:, :, :, polnum_data][selection]
        )
    )
    scale_factor_abs = np.sqrt(
        np.mean(
            np.abs(
                uvdata_reference_model.data_array[:, :, :, polnum_data][selection]
                / uvdata_deconv.data_array[:, :, :, polnum_data][selection]
            )
            ** 2.0
        )
    )
    scale_factor = scale_factor_abs * np.exp(1j * scale_factor_phase)
    uvdata_deconv.data_array[:, :, :, polnum_data] *= scale_factor

    polnum_gains = np.where(
        gains.jones_array == uvutils.polstr2num(polarization, x_orientation=uvdata_deconv.x_orientation)
    )[0][0]
    gains.gain_array[:, :, :, :, polnum_data] *= (scale_factor) ** -0.5


def tensorize_gains(uvcal, polarization, time_index, dtype=np.float32):
    """Helper function to extract gains into fitting tensors.

    Parameters
    ----------
    uvcal: UVCal object
        UVCal object holding gain data to tensorize.
    polarization: str
        pol-str of gain to extract.
    time_index: int
        index of time to extract.
    dtype: numpy.dtype
        dtype of tensors to output.

    Returns
    -------
    gains_re: tf.Tensor object.
        tensor object holding real component of gains
        for time_index and polarization
        shape is Nant x Nfreq
    gains_im: tf.Tensor object.
        tensor object holding imag component of gains
        for time_index and polarization
        shape is Nant x Nfreq

    """
    polnum = np.where(uvcal.jones_array == uvutils.polstr2num(polarization, x_orientation=uvcal.x_orientation))[0][0]
    gains_re = tf.convert_to_tensor(uvcal.gain_array[:, 0, :, time_index, polnum].squeeze().real, dtype=dtype)
    gains_im = tf.convert_to_tensor(uvcal.gain_array[:, 0, :, time_index, polnum].squeeze().imag, dtype=dtype)
    return gains_re, gains_im


def tensorize_pbl_model_comps_dictionary(fg_model_comps, dtype=np.float32):
    """Helper function generating mappings for per-baseline foreground modeling.

    Generates mappings between antenna pairs and foreground basis vectors accounting for redundancies.

    Parameters
    ----------
    fg_model_comps: dict with tuples of tuples of 2-tuples as keys and np.ndarrays as values.

    Returns
    -------
    fg_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.
    model_comp_tensor_map: diction with 2-tuples as keys an tf.Tensor objects as values.
        dictionary mapping antenna pairs to
    """
    model_comp_tensor_map = {}
    fg_range_map = {}
    startind = 0
    for grpnum, red_grp in enumerate(fg_model_comps.keys()):
        # set fg_range_map value based on first antenna-pair in redundant group.
        ncomponents = fg_model_comps[red_grp].shape[1]
        fg_range_map[red_grp[0][0]] = (startind, startind + ncomponents)
        startind += ncomponents
        for ap in red_grp[0]:
            model_comp_tensor_map[ap] = tf.convert_to_tensor(fg_model_comps[red_grp], dtype=dtype)
            fg_range_map[ap] = fg_range_map[red_grp[0][0]]
    return fg_range_map, model_comp_tensor_map


def yield_fg_model_tensor(fg_comps, fg_coeffs, nants, nfreqs):
    """Compute sparse tensor foreground model.

    Parameters
    ----------
    fg_comps: tf.sparse.SparseTensor or tf.Tensor object
        If tf.sparse.SparseTensor:
            Sparse tensor with dense shape of  (Nants^2 * Nfreqs) x Ncomponents
            Each column is a different data modeling component. The first axis ravels
            the data by ant1, ant2, freq. In the per-baseline
            modeling case, each column will be non-zero only over a single baseline.
        If tf.Tensor
            Tensor with shape Nants x Nants x Nfreqs x Ncomponents
    fg_coeffs: tf.Tensor object.
        if fg_comps is tf.sparse.SparseTensor:
            An Ncomponents x 1 tf.Tensor representing either the real or imag component
            of visibilities.
        if fg_comps is a tf.Tensor:
            An Ncomponents tf.Tensor representing either real or imag vis components.
    nants: int
        number of antennas in data to model.
    freqs: int
        number of frequencies in data to model.

    Returns
    -------
    model: tf.Tensor object
        nants x nants x nfreqs model of the visibility data
    """
    if isinstance(fg_comps, tf.sparse.SparseTensor):
        model = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps, fg_coeffs), (nants, nants, nfreqs))
    else:
        model = tf.reduce_sum(fg_comps * fg_coeffs, axis=3)
    return model


def yield_data_model_tensor(g_r, g_i, fg_comps, fg_r, fg_i, nants, nfreqs):
    """Compute an uncalibrated data model with gains and foreground coefficients.

    Parameters
    ----------
    g_r: tf.Tensor object
        real part of gains, Nants 1d tensor.
    g_i: tf.Tensor object
        imag part of gains, Nants 1d tensor.
    fg_comps: tf.Tensor or tf.sparse.SparseTensor object
        tf.sparse.SparseTensor object holding foreground modeling components
        (Nants^2 * Nfreqs) x Ncomponents
    fg_r: tf.Tensor object
        tf.Tensor object containing real components of foreground coefficients.
        Ncomponents x 1 tensor.
    fg_i: tf.Tensor object
        tf.Tensor object containing imag components of foreground coefficients
        Ncomponents x 1 tensor.
    nants: int
        number of modeled data antennas.
    nfreqs: int
        number of modeled frequenciers.

    Returns
    -------
    model_r: tf.Tensor object
        real component of data model
        3d: nants x nants x nfreqs
    model_i: tf.Tensor object
        imag component of data model
        3d: nants x nants x nfreqs
    """
    grgr = tf.einsum("ik,jk->ijk", g_r, g_r)
    gigi = tf.einsum("ik,jk->ijk", g_i, g_i)
    grgi = tf.einsum("ik,jk->ijk", g_r, g_i)
    gigr = tf.einsum("ik,jk->ijk", g_i, g_r)
    vr = yield_fg_model_tensor(fg_comps, fg_r, nants, nfreqs)
    vi = yield_fg_model_tensor(fg_comps, fg_i, nants, nfreqs)
    model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
    model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
    return model_r, model_i


def cal_loss_tensor(data_r, data_i, wgts, g_r, g_i, fg_model_comps, fg_r, fg_i, nants, nfreqs):
    """MSE Loss function for sparse tensor representation of visibilities.

    Parameters
    ----------
    data_r: tf.Tensor object.
        tf.Tensor object holding real parts of data
        3d: nants x nants x nfreqs
    data_i: tf.Tensor object.
        tf.Tensor object holding imag parts of data
        3d: nants x nants x nfreqs
    wgts: tf.Tensor object
        tf.Tensor object holding weights of each i,j,f visibility
        contribution to MSE loss.
    g_r: tf.Tensor object
        nants x nfreqs tf.Tensor holding real gain values.
    g_i: tf.Tensor object
        nants x nfreqs tf.Tensor holding imag gain values
    fg_comps: tf.sparse.SparseTensor object
        tf.sparse.SparseTensor object holding foreground modeling components
        (Nants^2 * Nfreqs) x Ncomponents
    fg_r: tf.Tensor object
        tf.Tensor object containing real components of foreground coefficients.
        Ncomponents x 1 tensor.
    fg_i: tf.Tensor object
        tf.Tensor object containing imag components of foreground coefficients
        Ncomponents x 1 tensor.
    nants: int
        number of modeled data antennas.
    nfreqs: int
        number of modeled frequenciers.

    Returns
    -------
    loss: scalar tf.Tensor object
        MSE loss between model given by g_r, g_i, fg_model_comps, fg_r, fg_i
        and data provided through data_r, data_i
    """
    model_r, model_i = yield_data_model_tensor(g_r, g_i, fg_model_comps, fg_r, fg_i, nants, nfreqs)
    return tf.reduce_sum(tf.square(data_r - model_r) * wgts + tf.square(data_i - model_i) * wgts)



def fit_gains_and_foregrounds(
    g_r,
    g_i,
    fg_r,
    fg_i,
    data_r=None,
    data_i=None,
    wgts=None,
    fg_comps=None,
    loss_function=None,
    use_min=False,
    tol=1e-14,
    maxsteps=10000,
    optimizer="Adamax",
    freeze_model=False,
    record_var_history=False,
    record_var_history_interval=1,
    verbose=False,
    notebook_progressbar=False,
    **opt_kwargs,
):
    """Run optimization loop to fit gains and foreground components.

    Parameters
    ----------
    g_r: tf.Tensor object.
        tf.Tensor object holding real parts of gains.
    g_i: tf.Tensor object.
        tf.Tensor object holding imag parts of gains.
    fg_r: tf.Tensor object.
        tf.Tensor object holding foreground coeffs.
    fg_i: tf.Tensor object.
        tf.Tensor object holding imag coeffs.
    loss_function: function
        loss function accepting g_r, g_i, fg_r, fg_i as arguments
        and returning a scalar tf.Tensor object (loss) as output.
    use_min: bool, optional
        if True, use the value that minimizes the loss function
        regardless of where optimization loop ended up
        (prevents overshooting due to excess momentum)
    tol: float, optional
        halt optimization loop once the loss changes by less then this value.
        default is 1e-14
    maxsteps: int, optional
        maximum number of opt.minimize calls before halting.
        default is 10000
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary which contains optimizers described in
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        default is 'Adamax'
    freeze_model: bool, optional
        Only optimize loss function wrt gain variables. This is effectively traditional model-based calibration
        with sky_model as the model (but projected onto the foreground basis vectors).
        default is False.
    record_var_history: bool, optional
        record detailed history of all g_r and g_i values.
        default is True.
    record_var_history_interval: int, optional
        interval of otpimization steps to store detailed history
        default is 1.
    verbose: bool, optional
        lots of text output
        default is False.
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    opt_kwargs: kwarg dict
        additional kwargs for tf.opt.Optimizer(). See tensorflow docs.

    Returns
    -------
    g_r_opt: tf.Tensor object
        real part of optimized gains.
    g_i_opt: tf.Tensor object
        imag part of optimized gains.
    fg_r_opt: tf.Tensor object
        real part of foreground coeffs.
    fg_i_opt: tf.Tensor object.
        imag part of optimized foreground coeffs.
    fit_history: dict
        dictionary containing fit history for each time-step and polarization in the data with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coeffs
            'fg_i': imag part of foreground coeffs
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    g_r = tf.Variable(g_r)
    g_i = tf.Variable(g_i)
    if not freeze_model:
        fg_r = tf.Variable(fg_r)
        fg_i = tf.Variable(fg_i)
    # initialize the optimizer.
    opt = OPTIMIZERS[optimizer](**opt_kwargs)
    # set up history recording
    fit_history = {"loss": []}
    if record_var_history:
        fit_history["g_r"] = []
        fit_history["g_i"] = []
        if not freeze_model:
            fit_history["fg_r"] = []
            fit_history["fg_i"] = []

    # perform optimization loop.
    if freeze_model:
        vars = [g_r, g_i]
    else:
        vars = [g_r, g_i, fg_r, fg_i]
    min_loss = 9e99
    echo(
        f"{datetime.datetime.now()} Building Computational Graph...\n",
        verbose=verbose,
    )
    nants = g_r.shape[0]
    nfreqs = g_r.shape[1]
    if loss_function is not None:
        @tf.function
        def cal_loss():
            return loss_function(g_r=g_r, g_i=g_i, fg_i=fg_i, fg_r=fg_r)
    else:
        if isinstance(fg_comps, tf.Tensor):
            def cal_loss():
                grgr = tf.einsum('ik,jk->ijk',gr, gr)
                gigi = tf.einsum('ik,jk->ijk',gi, gi)
                grgi = tf.einsum('ik,jk->ijk',gr, gi)
                gigr = tf.einsum('ik,jk->ijk',gi, gr)
                vr = tf.reduce_sum(fg_comps * fg_r, axis=3)
                vi = tf.reduce_sum(fg_comps * fg_i, axis=3)
                model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
                model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
                return tf.reduce_sum(tf.square(data_r - model_r) * wgts + tf.square(data_i - model_i) * wgts)
        elif isinstance(fg_comps, tf.sparse.SparseTensor):
            def cal_loss():
                grgr = tf.einsum('ik,jk->ijk',gr, gr)
                gigi = tf.einsum('ik,jk->ijk',gi, gi)
                grgi = tf.einsum('ik,jk->ijk',gr, gi)
                gigr = tf.einsum('ik,jk->ijk',gi, gr)
                model_r = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps, fg_r), (nants, nants, nfreqs))
                model_i = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps, fg_i), (nants, nants, nfreqs))
                model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
                model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
                return tf.reduce_sum(tf.square(data_r - model_r) * wgts + tf.square(data_i - model_i) * wgts)
    loss_i = cal_loss().numpy()
    echo(
        f"{datetime.datetime.now()} Performing Gradient Descent. Initial MSE of {loss_i:.2e}...\n",
        verbose=verbose,
    )
    for step in PBARS[notebook_progressbar](range(maxsteps)):
        with tf.GradientTape() as tape:
            loss = cal_loss()
        grads = tape.gradient(loss, vars)
        opt.apply_gradients(zip(grads, vars))
        fit_history["loss"].append(loss.numpy())
        if record_var_history and step % record_var_history_interval == 0:
            fit_history["g_r"].append(g_r.numpy())
            fit_history["g_i"].append(g_i.numpy())
            if not freeze_model:
                fit_history["fg_r"].append(fg_r.numpy())
                fit_history["fg_i"].append(fg_i.numpy())
        if use_min and fit_history["loss"][-1] < min_loss:
            # store the g_r, g_i, fg_r, fg_i values that minimize loss
            # in case of overshoot.
            min_loss = fit_history["loss"][-1]
            g_r_opt = g_r.value()
            g_i_opt = g_i.value()
            if not freeze_model:
                fg_r_opt = fg_r.value()
                fg_i_opt = fg_i.value()
            else:
                fg_r_opt = fg_r
                fg_i_opt = fg_i

        if step >= 1 and np.abs(fit_history["loss"][-1] - fit_history["loss"][-2]) < tol:
            echo(
                f"Tolerance thresshold met with delta of {np.abs(fit_history['loss'][-1] - fit_history['loss'][-2]):.2e}. Terminating...\n ",
                verbose=verbose,
            )
            break

    # if we dont use use_min, then the last
    # visited set of parameters will be used
    # to set the ML params.
    if not use_min:
        min_loss = fit_history["loss"][-1]
        g_r_opt = g_r.value()
        g_i_opt = g_i.value()
        if not freeze_model:
            fg_r_opt = fg_r.value()
            fg_i_opt = fg_i.value()
        else:
            fg_r_opt = fg_r
            fg_i_opt = fg_i
    echo(
        f"{datetime.datetime.now()} Finished Gradient Descent. MSE of {min_loss:.2e}...\n",
        verbose=verbose,
    )
    return g_r_opt, g_i_opt, fg_r_opt, fg_i_opt, fit_history


def yield_fg_model_pbl_dictionary_method(
    i,
    j,
    fg_coeffs_re,
    fg_coeffs_im,
    fg_range_map,
    components_map,
):
    """Helper function for retrieving a per-baseline foreground model using the dictionary mapping technique

    From empirical experimentation, this technique works best in graph mode on CPUs. We recommend
    the array method if working with GPUs.

    Parameters
    ----------
    i: int
        i correlation index
    j: int
        j correlation index
    fg_coeffs_re: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the real components of coeffs multiplying foreground
        basis vectors.
    fg_coeffs_im: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the imaginary components of coeffs multiplying foreground
        basis vectors.
    fg_range_map: dict with 2-tuple int keys and 2-int tuple values
        dictionary with keys that are (i, j) pairs of correlation indices which map to
        integer 2-tuples (index_low, index_high) representing the lower and upper indices of
        the fg_coeffs tensor. Lower index is inclusive, upper index is exclusive
        (consistent with python indexing convention).
    components_map: dict of 2-tuple int keys and tf.Tensor object values.
        dictionary with keys that are (i, j) integer pairs of correlation indices which map to
        Nfreq x Nforeground tensorflow tensor where each column is a separate per-baseline
        foreground basis component.

    Returns
    -------
    fg_model_re: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the real part of the (i, j) correlation.
    fg_model_im: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the imag part of the (i, j) correlation

    """
    fg_model_re = tf.reduce_sum(
        components_map[i, j] * fg_coeffs_re[fg_range_map[(i, j)][0] : fg_range_map[(i, j)][1]],
        axis=1,
    )  # real part of fg model.
    fg_model_im = tf.reduce_sum(
        components_map[i, j] * fg_coeffs_im[fg_range_map[(i, j)][0] : fg_range_map[(i, j)][1]],
        axis=1,
    )  # imag part of fg model.
    return fg_model_re, fg_model_im


def insert_model_into_uvdata_tensor(
    uvdata,
    time_index,
    polarization,
    ants_map,
    red_grps,
    model_r,
    model_i,
    scale_factor=1.0,
):
    """Insert fitted tensor values back into uvdata object for sparse tensor mode.

    Parameters
    ----------
    uvdata: UVData object
        uvdata object to insert model data into.
    time_index: int
        time index to insert model data at.
    polarization: str
        polarization to insert.
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    red_grps: list of lists of int 2-tuples
        a list of lists of 2-tuples where all antenna pairs within each sublist
        are redundant with eachother. Assumes that conjugates are correctly taken.
    model_r: tf.Tensor object
        an Nants_data x Nants_data x Nfreqs tf.Tensor object with real parts of data
    model_i: tf.Tensor object
        an Nants_data x Nants_data x Nfreqs tf.Tensor object with imag parts of model
    scale_factor: float, optional
        overall scaling factor to divide tensorized data by.
        default is 1.0

    Returns
    -------
    N/A: Modifies uvdata inplace.

    """
    antpairs_data = uvdata.get_antpairs()
    polnum = np.where(
        uvdata.polarization_array == uvutils.polstr2num(polarization, x_orientation=uvdata.x_orientation)
    )[0][0]
    for red_grp in red_grps:
        for ap in red_grp:
            i, j = ants_map[ap[0]], ants_map[ap[1]]
            if ap in antpairs_data:
                dinds = uvdata.antpair2ind(ap)[time_index]
                model = model_r[i, j].numpy() + 1j * model_i[i, j].numpy()
            else:
                dinds = uvdata.antpair2ind(ap[::-1])[time_index]
                model = model_r[i, j].numpy() - 1j * model_i[i, j].numpy()
            uvdata.data_array[dinds, 0, :, polnum] = model * scale_factor


def insert_model_into_uvdata_dictionary(
    uvdata,
    time_index,
    polarization,
    fg_range_map,
    model_comps_map,
    fg_coeffs_re,
    fg_coeffs_im,
    scale_factor=1.0,
):
    """Insert tensor values back into uvdata object for dictionary mode.

    Parameters
    ----------
    uvdata: UVData object
        uvdata object to insert model data into.
    time_index: int
        time index to insert model data at.
    polarization: str
        polarization to insert.
    fg_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline model component number.
    components_map: dict of 2-tuple int keys and tf.Tensor object values.
        dictionary with keys that are (i, j) integer pairs of correlation indices which map to
        Nfreq x Nforeground tensorflow tensor where each column is a separate per-baseline
        foreground basis component.
    fg_coeffs_re: tf.Tensor object
        1d tensor containing real parts of coeffs for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    fg_coeffs_im: tf.Tensor object
        1d tensor containing imag parts of coeffs for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    scale_factor: float
        scale factor that model was divided by for fitting.
        default is 1.

    Returns
    -------
    N/A. Modifies uvdata in-place.
    """
    data_antpairs = uvdata.get_antpairs()
    polnum = np.where(
        uvdata.polarization_array == uvutils.polstr2num(polarization, x_orientation=uvdata.x_orientation)
    )[0][0]
    for ap in fg_range_map:
        fgrange = slice(fg_range_map[ap][0], fg_range_map[ap][1])
        model = model_comps_map[ap].numpy() @ (fg_coeffs_re.numpy()[fgrange] + 1j * fg_coeffs_im.numpy()[fgrange])
        model *= scale_factor
        if ap in data_antpairs:
            dinds = uvdata.antpair2ind(ap)[time_index]
        else:
            dinds = uvdata.antpair2ind(ap[::-1])[time_index]
            model = np.conj(model)
        uvdata.data_array[dinds, 0, :, polnum] = model


def insert_gains_into_uvcal(uvcal, time_index, polarization, gains_re, gains_im):
    """Insert tensorized gains back into uvcal object

    Parameters
    ----------
    uvdata: UVData object
        uvdata object to insert model data into.
    time_index: int
        time index to insert model data at.
    polarization: str
        polarization to insert.
    gains_re: dict with int keys and tf.Tensor object values
        dictionary mapping i antenna numbers to Nfreq 1d tf.Tensor object
        representing the real component of the complex gain for antenna i.
    gains_im: dict with int keys and tf.Tensor object values
        dictionary mapping j antenna numbers to Nfreq 1d tf.Tensor object
        representing the imag component of the complex gain for antenna j.

    Returns
    -------
    N/A: Modifies uvcal inplace.
    """
    polnum = np.where(uvcal.jones_array == uvutils.polstr2num(polarization, x_orientation=uvcal.x_orientation))[0][0]
    for ant_index in range(uvcal.Nants_data):
        uvcal.gain_array[ant_index, 0, :, time_index, polnum] = (
            gains_re[ant_index].numpy() + 1j * gains_im[ant_index].numpy()
        )


# get the calibrated model
def yield_data_model_pbl_dictionary(
    i,
    j,
    gains_re,
    gains_im,
    ants_map,
    fg_coeffs_re,
    fg_coeffs_im,
    fg_range_map,
    components_map,
):
    """Helper function for retrieving a per-baseline uncalibrted foreground model using the dictionary mapping technique

    From empirical experimentation, this technique works best in graph mode on CPUs. We recommend
    the array method if working with GPUs.

    Parameters
    ----------
    i: int
        i correlation index
    j: int
        j correlation index
    gains_re: dict with int keys and tf.Tensor object values
        dictionary mapping i antenna numbers to Nfreq 1d tf.Tensor object
        representing the real component of the complex gain for antenna i.
    gains_im: dict with int keys and tf.Tensor object values
        dictionary mapping j antenna numbers to Nfreq 1d tf.Tensor object
        representing the imag component of the complex gain for antenna j.
    ants_map: dict mapping integer keys to integer values.
        dictionary mapping antenna number to antenna index in gains_re and gains_im.
    fg_coeffs_re: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the real components of coeffs multiplying foreground
        basis vectors.
    fg_coeffs_im: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the imaginary components of coeffs multiplying foreground
        basis vectors.
    fg_range_map: dict with 2-tuple int keys and 2-int tuple values
        dictionary with keys that are (i, j) pairs of correlation indices which map to
        integer 2-tuples (index_low, index_high) representing the lower and upper indices of
        the fg_coeffs tensor. Lower index is inclusive, upper index is exclusive
        (consistent with python indexing convention).
    components_map: dict of 2-tuple int keys and tf.Tensor object values.
        dictionary with keys that are (i, j) integer pairs of correlation indices which map to
        Nfreq x Nforeground tensorflow tensor where each column is a separate per-baseline
        foreground basis component.

    Returns
    -------
    uncal_model_re: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the real part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    uncal_model_im: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the imag part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    """
    (fg_model_re, fg_model_im,) = yield_fg_model_pbl_dictionary_method(
        i,
        j,
        fg_coeffs_re=fg_coeffs_re,
        fg_coeffs_im=fg_coeffs_im,
        fg_range_map=fg_range_map,
        components_map=components_map,
    )
    i, j = ants_map[i], ants_map[j]
    uncal_model_re = (gains_re[i] * gains_re[j] + gains_im[i] * gains_im[j]) * fg_model_re + (
        gains_re[i] * gains_im[j] - gains_im[i] * gains_re[j]
    ) * fg_model_im  # real part of model with gains
    uncal_model_im = (gains_re[i] * gains_re[j] + gains_im[i] * gains_im[j]) * fg_model_im + (
        gains_im[i] * gains_re[j] - gains_re[i] * gains_im[j]
    ) * fg_model_re  # imag part of model with gains
    return uncal_model_re, uncal_model_im


def cal_loss_dictionary(
    gains_re,
    gains_im,
    fg_coeffs_re,
    fg_coeffs_im,
    data_re,
    data_im,
    wgts,
    ants_map,
    fg_range_map,
    components_map,
):
    """MSE loss-function for dictionary method of computing data model.

    Parameters
    ----------
    gains_re: dictionary with ints as keys and tf.Tensor objects as values.
        dictionary mapping antenna numbers to Nfreq 1d tensors representing the
        real part of the model for each antenna.
    gains_im: dictionary with ints as keys and tf.Tensor objects as values.
        dictionary mapping antenna numbers to Nfreq 1d tensors representing the
        imag part of the model for each antenna.
    fg_coeffs_re: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to N-foreground-coeff 1d tensors representing the
        real part of the model for each antenna.
    fg_coeffs_im: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to N-foreground-coeff 1d tensors representing the
        imag part of the model for each antenna.
    data_re: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        the real part of the target data for each baseline.
    data_im: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        the imag part of the target data for each baseline.
    wgts: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        per-frequency weights for each baseline contributing to the loss function.
    ants_map: dict mapping integer keys to integer values.
        dictionary mapping antenna number to antenna index in gains_re and gains_im.
    fg_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.

    Returns
    -------
    loss_total: tf.Tensor scalar
        The MSE (mean-squared-error) loss value for the input model parameters and data.
    """
    loss_total = 0.0
    for i, j in fg_range_map:
        model_r, model_i = yield_data_model_pbl_dictionary(
            i,
            j,
            gains_re=gains_re,
            gains_im=gains_im,
            ants_map=ants_map,
            fg_coeffs_re=fg_coeffs_re,
            fg_coeffs_im=fg_coeffs_im,
            fg_range_map=fg_range_map,
            components_map=components_map,
        )
        loss_total += tf.reduce_sum(tf.square(model_r - data_re[i, j]) * wgts[i, j])
        # imag part of loss
        loss_total += tf.reduce_sum(tf.square(model_i - data_im[i, j]) * wgts[i, j])
    return loss_total


def tensorize_pbl_data_dictionary(
    uvdata,
    polarization,
    time_index,
    red_grps,
    scale_factor=1.0,
    weights=None,
    dtype=np.float32,
):
    """extract data from uvdata object into a dict of tensors for fitting.

    Produce dictionaries of data tensors to be used in fitting for the dictionary
    variant which works best on CPUs.

    Parameters
    ----------
    uvdata: UVData object.
        UVData object with data to extract tensors from.
    polarization: str.
        String encoding polarization to extract from data.
    time_index: int.
        integer index of time to extract (assumes times sorted ascending).
    red_grps: list of lists of int 2-tuples
        list of lists where each sublist is a redundant group with antenna
        pairs ordered such that there are no conjugates of eachother within
        each group.
    data_scale_factor: float, optional
        overall scaling factor to divide tensorized data by.
        default is 1.0
    dtype: np.dtype
        data type to store tensorized data in.

    Returns
    -------
    data_re: dict with int 2-tuple keys and tf.Tensor values
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each
        representing the real part of the spectum for baseline (i, j) with pol
        polarization at time-index time_index.
    data_im: dict with int 2-tuple keys and tf.Tensor values
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each
        representing the imag part of the spectum for baseline (i, j) with pol
        polarization at time-index time_index.
    wgts: dict with int 2-tuple keys and tf.Tensor values.
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each representing
        the real weigths

    """
    data_re = {}
    data_im = {}
    wgts = {}
    wgtsum = 0.0
    for red_grp in red_grps:
        for ap in red_grp:
            bl = ap + (polarization,)
            data_re[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].real / scale_factor, dtype=dtype)
            data_im[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].imag / scale_factor, dtype=dtype)
            if weights is None:
                wgts[ap] = tf.convert_to_tensor(
                    ~uvdata.get_flags(bl)[time_index] * uvdata.get_nsamples(bl)[time_index], dtype=dtype
                )
            else:
                if ap in weights.get_antpairs():
                    dinds = weights.antpair2ind(*ap)
                else:
                    dinds = weights.antpair2ind(*ap[::-1])
                polnum = np.where(
                    weights.polarization_array == uvutils.polstr2num(polarization, x_orientation=weights.x_orientation)
                )[0][0]
                wgts[ap] = weights.weights_array[dinds, 0, :, polnum].astype(dtype) * ~uvdata.get_flags(bl)[time_index]
                wgts[ap] = tf.convert_to_tensor(wgts[ap], dtype=dtype)
            wgtsum += np.sum(wgts[ap])
    for ap in wgts:
        wgts[ap] /= wgtsum

    return data_re, data_im, wgts


def tensorize_fg_coeffs(
    uvdata,
    model_component_dict,
    time_index,
    polarization,
    scale_factor=1.0,
    force2d=False,
    dtype=np.float32,
):
    """Initialize foreground coefficient tensors from uvdata and modeling component dictionaries.


    Parameters
    ----------
    uvdata: UVData object.
        UVData object holding model data.
    model_component_dict: dict with tuples of tuples of int 2-tuples as keys keys and numpy.ndarray values.
        dictionary holding int 2-tuple keys mapping to Nfreq x Nfg ndarrrays
        used to model each individual baseline.
    red_grps: list of lists of int 2-tuples
        lists of redundant baseline groups with antenna pairs set to avoid conjugation.
    time_index: int
        time index of data to calculate foreground coeffs for.
    polarization: str
        polarization to calculate foreground coeffs for.
    scale_factor: float, optional
        factor to scale data by.
        default is 1.
    force2d: bool, optional
        if True, add additional dummy dimension to make 2d
        this is necessary for sparse matrix representation.
        default is False.
    dtype: numpy.dtype
        data type to store tensors.

    Returns
    -------
    fg_coeffs_re: tf.Tensor object
        1d tensor containing real parts of coeffs for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    fg_coeffs_im: tf.Tensor object
        1d tensor containing imag parts of coeffs for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    """
    fg_coeffs_re = []
    fg_coeffs_im = []
    for fit_grp in model_component_dict:
        fg_coeff = 0.0
        blnum = 0
        for red_grp in fit_grp:
            for ap in red_grp:
                bl = ap + (polarization,)
                fg_coeff += (
                    uvdata.get_data(bl)[time_index] * ~uvdata.get_flags(bl)[time_index]
                ) @ model_component_dict[fit_grp][blnum * uvdata.Nfreqs : (blnum + 1) * uvdata.Nfreqs]
            blnum += 1
        fg_coeffs_re.extend(fg_coeff.real)
        fg_coeffs_im.extend(fg_coeff.imag)

    fg_coeffs_re = np.asarray(fg_coeffs_re) / scale_factor
    fg_coeffs_im = np.asarray(fg_coeffs_im) / scale_factor
    if force2d:
        fg_coeffs_re = fg_coeffs_re.reshape((len(fg_coeffs_re), 1))
        fg_coeffs_im = fg_coeffs_im.reshape((len(fg_coeffs_im), 1))
    fg_coeffs_re = tf.convert_to_tensor(fg_coeffs_re, dtype=dtype)
    fg_coeffs_im = tf.convert_to_tensor(fg_coeffs_im, dtype=dtype)

    return fg_coeffs_re, fg_coeffs_im


def calibrate_and_model_tensor(
    uvdata,
    fg_model_comps,
    gains=None,
    freeze_model=False,
    optimizer="Adamax",
    tol=1e-14,
    maxsteps=10000,
    include_autos=False,
    verbose=False,
    sky_model=None,
    dtype=np.float32,
    use_min=False,
    record_var_history=False,
    record_var_history_interval=1,
    use_redundancy=False,
    notebook_progressbar=False,
    correct_resid=False,
    correct_model=False,
    weights=None,
    sparse_threshold=1e-1,
    **opt_kwargs,
):
    """Perform simultaneous calibration and foreground fitting using sparse tensors.

    This method should be used in place of calibrate_and_model_pbl_dictionary_method
    when using GPUs while the latter tends to perform better on CPUs.

    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    fg_model_comps: dictionary
        dictionary with keys that are tuples of tuples of 2-tuples (thats right, 3 levels)
        in the first level, each tuple represents a 'modeling group' visibilities in each
        modeling group are represented by a set of basis vectors that span all baselines in that
        group with elements raveled by baseline and then frequency. Each tuple in the modeling group is a
        'redundant group' representing visibilities that we will represent with identical component coefficients
        each element of each 'redundant group' is a 2-tuple antenna pair. Our formalism easily accomodates modeling
        visibilities as redundant or non redundant (one simply needs to make each redundant group length 1).
        values are real numpy arrays with size (Ngrp * Nfreqs) * Ncomponents
    gains: UVCal object
        UVCal with initial gain estimates.
        There many smart ways to obtain initial gain estimates
        but this is beyond the scope of calamity (for example, firstcal, logcal, sky-based cal).
        Users can determine initial gains with their favorite established cal algorithm.
        default is None -> start with unity gains.
        WARNING: At the present, the flags in gains are not propagated/used! Make sure flags in uvdata object!
    freeze_model: bool, optional
        Only optimize loss function wrt gain variables. This is effectively traditional model-based calibration
        with sky_model as the model (but projected onto the foreground basis vectors).
        default is False.
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary which contains optimizers described in
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        default is 'Adamax'
    tol: float, optional
        halting condition for optimizer loop. Stop loop when the change in the cost function falls
        below tol.
        default is 1e-14
    maxsteps: int, optional
        maximum number of opt.minimize calls before halting.
        default is 10000
    include_autos: bool, optional
        include autocorrelations in fitting.
        default is False.
    verbose: bool, optional
        generate lots of text.
        default is False.
    sky_model: UVData object, optional
        a sky-model to use for initial estimates of foreground coeffs and
        to set overall flux scale and phases.
        Note that this model is not used to obtain initial gain estimates.
        These must be provided through the gains argument.
    dtype: numpy dtype, optional
        the float precision to be used in tensorflow gradient descent.
        runtime scales roughly inversely linear with precision.
        default is np.float32
    use_min: bool, optional
        If True, use the set of parameters that determine minimum as the ML params
        If False, use the last set of parameters visited by the optimization loop.
    record_var_history: bool, optional
        keep detailed record of optimization history of variables.
        default is False.
    record_var_history_interval: int optional
        store var history in detailed record every record_var_history_interval steps.
        default is 1.
    use_redundancy: bool, optional
        if true, solve for one set of foreground coeffs per redundant baseline group
        instead of per baseline.
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    red_tol: float, optional
        tolerance for determining baselines redundant (meters)
        default is 1.0
    correct_resid: bool, optional
        if True, gain correct residual.
        default is False
    correct_model: bool, optional
        if True, gain correct model.
        default is False
    weights: UVFlag object, optional.
        UVFlag weights object containing weights to use for data fitting.
        default is None -> use nsamples * ~flags
    sparse_threshold: float, optional
        if fraction of elements in foreground modeling vector matrix that are non-zero
        is greater then this value, then use a dense representation.
        Otherwise use a sparse representation.
        default is 1e-1
    opt_kwargs: kwarg_dict
        kwargs for tf.optimizers

    Returns
    -------
    model: UVData object
        uvdata object containing model of the foregrounds
    resid: UVData object
        uvdata object containing resids which are the data minus
        the model with gains multiplied and then with the gains divided out.
    gains: UVCal object
        uvcal object containing estimates of the gain solutions. These solutions
        are not referenced to any sky model and are likely orders of
    fit_history:
        dictionary containing fit history with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coeffs
            'fg_i': imag part of foreground coeffs
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    antpairs_data = uvdata.get_antpairs()
    if not include_autos:
        antpairs_data = set([ap for ap in antpairs_data if ap[0] != ap[1]])
    red_grps = []
    # generate redundant groups
    for fit_grp in fg_model_comps.keys():
        for red_grp in fit_grp:
            red_grps.append(red_grp)
    uvdata = uvdata.select(inplace=False, bls=[ap for ap in antpairs_data])
    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)
    if gains is None:
        echo(
            f"{datetime.datetime.now()} Gains are None. Initializing gains starting with unity...\n",
            verbose=verbose,
        )
        gains = cal_utils.blank_uvcal_from_uvdata(uvdata)

    if sky_model is None:
        echo(
            f"{datetime.datetime.now()} Sky model is None. Initializing from data...\n",
            verbose=verbose,
        )
        sky_model = cal_utils.apply_gains(uvdata, gains)
    sky_model = sky_model.select(inplace=False, bls=[ap for ap in antpairs_data])
    fit_history = {}
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    # generate sparse tensor to hold foreground components.
    fg_comp_tensor = tensorize_fg_model_comps(
        fg_model_comps=fg_model_comps,
        ants_map=ants_map,
        dtype=dtype,
        nfreqs=sky_model.Nfreqs,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        sparse_threshold=sparse_threshold,
    )
    echo(
        f"{datetime.datetime.now()}Finished Computing sparse foreground components matrix...\n",
        verbose=verbose,
    )

    # loop through polarization and times.
    for polnum, pol in enumerate(uvdata.get_pols()):
        echo(
            f"{datetime.datetime.now()} Working on pol {pol}, {polnum + 1} of {uvdata.Npols}...\n",
            verbose=verbose,
        )
        fit_history_p = {}
        for time_index in range(uvdata.Ntimes):
            rmsdata = np.sqrt(
                np.mean(
                    np.abs(
                        uvdata.data_array[time_index :: uvdata.Ntimes, 0, :, polnum][
                            ~uvdata.flag_array[time_index :: uvdata.Ntimes, 0, :, polnum]
                        ]
                    )
                    ** 2.0
                )
            )
            echo(
                f"{datetime.datetime.now()} Working on time {time_index + 1} of {uvdata.Ntimes}...\n",
                verbose=verbose,
            )
            echo(f"{datetime.datetime.now()} Tensorizing data...\n", verbose=verbose)
            data_r, data_i, wgts = tensorize_data(
                uvdata,
                red_grps=red_grps,
                ants_map=ants_map,
                polarization=pol,
                time_index=time_index,
                data_scale_factor=rmsdata,
                weights=weights,
            )
            echo(f"{datetime.datetime.now()} Tensorizing Gains...\n", verbose=verbose)
            gains_r, gains_i = tensorize_gains(gains, dtype=dtype, time_index=time_index, polarization=pol)
            # generate initial guess for foreground coeffs.
            echo(
                f"{datetime.datetime.now()} Tensorizing Foreground coeffs...\n",
                verbose=verbose,
            )
            fg_r, fg_i = tensorize_fg_coeffs(
                uvdata=sky_model,
                model_component_dict=fg_model_comps,
                dtype=dtype,
                time_index=time_index,
                polarization=pol,
                scale_factor=rmsdata,
                force2d=isinstance(fg_comp_tensor, tf.sparse.SparseTensor),
            )

            cal_loss = lambda g_r, g_i, fg_r, fg_i: cal_loss_tensor(
                data_r=data_r,
                data_i=data_i,
                wgts=wgts,
                g_r=g_r,
                g_i=g_i,
                fg_r=fg_r,
                fg_i=fg_i,
                fg_model_comps=fg_comp_tensor,
                nants=uvdata.Nants_data,
                nfreqs=uvdata.Nfreqs,
            )
            # derive optimal gains and foregrounds
            (gains_r, gains_i, fg_r, fg_i, fit_history_p[time_index],) = fit_gains_and_foregrounds(
                g_r=gains_r,
                g_i=gains_i,
                fg_r=fg_r,
                fg_i=fg_i,
                data_r=data_r,
                data_i=data_i,
                wgts=wgts,
                #loss_function=cal_loss,
                record_var_history=record_var_history,
                record_var_history_interval=record_var_history_interval,
                optimizer=optimizer,
                use_min=use_min,
                freeze_model=freeze_model,
                notebook_progressbar=notebook_progressbar,
                verbose=verbose,
                tol=tol,
                maxsteps=maxsteps,
                **opt_kwargs,
            )
            # insert into model uvdata.
            insert_model_into_uvdata_tensor(
                uvdata=model,
                time_index=time_index,
                polarization=pol,
                ants_map=ants_map,
                red_grps=red_grps,
                model_r=yield_fg_model_tensor(fg_comp_tensor, fg_r, uvdata.Nants_data, uvdata.Nfreqs),
                model_i=yield_fg_model_tensor(fg_comp_tensor, fg_i, uvdata.Nants_data, uvdata.Nfreqs),
                scale_factor=rmsdata,
            )
            # insert gains into uvcal
            insert_gains_into_uvcal(
                uvcal=gains,
                time_index=time_index,
                polarization=pol,
                gains_re=gains_r,
                gains_im=gains_i,
            )
        fit_history[polnum] = fit_history_p
        if not freeze_model:
            renormalize(
                uvdata_reference_model=sky_model,
                uvdata_deconv=model,
                gains=gains,
                polarization=pol,
                uvdata_flags=uvdata,
            )
    model_with_gains = cal_utils.apply_gains(model, gains, inverse=True)
    if not correct_model:
        model = model_with_gains
    resid.data_array -= model_with_gains.data_array
    if correct_resid:
        resid = cal_utils.apply_gains(resid, gains)

    return model, resid, gains, fit_history


def calibrate_and_model_pbl_dictionary_method(
    uvdata,
    fg_model_comps,
    gains=None,
    freeze_model=False,
    optimizer="Adamax",
    tol=1e-14,
    maxsteps=10000,
    include_autos=False,
    verbose=False,
    sky_model=None,
    dtype=np.float32,
    use_min=False,
    record_var_history=False,
    record_var_history_interval=1,
    use_redundancy=False,
    notebook_progressbar=False,
    red_tol=1.0,
    correct_resid=False,
    correct_model=False,
    weights=None,
    **opt_kwargs,
):
    """Perform simultaneous calibration and fitting of foregrounds using method that loops over baselines.

    This approach gives up on trying to invert the wedge but can be used on practically any array.
    Use this in place of calibrate_and_model_tensor when working on CPUs but
    use the latter with GPUs.

    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    fg_model_comps: dictionary
        dictionary containing Nfreq x Nbasis design matrices
        describing the basis vectors being used to model each baseline with keys corresponding
        antenna pairs.
    gains: UVCal object
        UVCal with initial gain estimates.
        There many smart ways to obtain initial gain estimates
        but this is beyond the scope of calamity (for example, firstcal, logcal, sky-based cal).
        Users can determine initial gains with their favorite established cal algorithm.
        default is None -> start with unity gains.
        WARNING: At the present, the flags in gains are not propagated/used! Make sure flags in uvdata object!
    freeze_model: bool, optional
        Only optimize loss function wrt gain variables. This is effectively traditional model-based calibration
        with sky_model as the model (but projected onto the foreground basis vectors).
        default is False.
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary which contains optimizers described in
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        default is 'Adamax'
    tol: float, optional
        halting condition for optimizer loop. Stop loop when the change in the cost function falls
        below tol.
        default is 1e-14
    maxsteps: int, optional
        maximum number of opt.minimize calls before halting.
        default is 10000
    include_autos: bool, optional
        include autocorrelations in fitting.
        default is False.
    verbose: bool, optional
        generate lots of text.
        default is False.
    sky_model: UVData object, optional
        a sky-model to use for initial estimates of foreground coeffs and
        to set overall flux scale and phases.
        Note that this model is not used to obtain initial gain estimates.
        These must be provided through the gains argument.
    dtype: numpy dtype, optional
        the float precision to be used in tensorflow gradient descent.
        runtime scales roughly inversely linear with precision.
        default is np.float32
    use_min: bool, optional
        If True, use the set of parameters that determine minimum as the ML params
        If False, use the last set of parameters visited by the optimization loop.
    record_var_history: bool, optional
        keep detailed record of optimization history of variables.
        default is False.
    record_var_history_interval: int optional
        store var history in detailed record every record_var_history_interval steps.
        default is 1.
    use_redundancy: bool, optional
        if true, solve for one set of foreground coeffs per redundant baseline group
        instead of per baseline.
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    red_tol: float, optional
        tolerance for treating baselines as redundant (meters)
        default is 1.0
    correct_resid: bool, optional
        if True, gain correct residual.
        default is False
    correct_model: bool, optional
        if True, gain correct model.
        default is False
    weights: UVFlag object, optional.
        UVFlag weights object containing weights to use for data fitting.
        default is None -> use nsamples * ~flags

    Returns
    -------
    model: UVData object
        uvdata object containing model of the foregrounds
    resid: UVData object
        uvdata object containing resids which are the data minus
        the model with gains multiplied and then with the gains divided out.
    gains: UVCal object
        uvcal object containing estimates of the gain solutions. These solutions
        are not referenced to any sky model and are likely orders of
    fit_history:
        dictionary containing fit history with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coeffs
            'fg_i': imag part of foreground coeffs
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    antpairs_data = uvdata.get_antpairs()
    if not include_autos:
        antpairs_data = set([ap for ap in antpairs_data if ap[0] != ap[1]])
    red_grps = []
    # generate redundant groups
    for fit_grp in fg_model_comps.keys():
        for red_grp in fit_grp:
            red_grps.append(red_grp)
    uvdata = uvdata.select(inplace=False, bls=[ap for ap in antpairs_data])
    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)
    if gains is None:
        echo(
            f"{datetime.datetime.now()} Gains are None. Initializing gains starting with unity...\n",
            verbose=verbose,
        )
        gains = cal_utils.blank_uvcal_from_uvdata(uvdata)
    if sky_model is None:
        echo(
            f"{datetime.datetime.now()} Sky model is None. Initializing from data...\n",
            verbose=verbose,
        )
        sky_model = cal_utils.apply_gains(uvdata, gains)
    sky_model = sky_model.select(inplace=False, bls=[ap for ap in antpairs_data])
    fit_history = {}
    echo(
        f"{datetime.datetime.now()} Generating map between antenna pairs and modeling vectors...\n",
        verbose=verbose,
    )
    (
        fg_range_map,
        model_comps_map,
    ) = tensorize_pbl_model_comps_dictionary(fg_model_comps, dtype=dtype)
    # index antenna numbers (in gain vectors).
    ants_map = {gains.antenna_numbers[i]: i for i in range(len(gains.antenna_numbers))}
    # We do fitting per time and per polarization and time.
    for polnum, pol in enumerate(uvdata.get_pols()):
        echo(
            f"{datetime.datetime.now()} Working on pol {pol}, {polnum + 1} of {uvdata.Npols}...\n",
            verbose=verbose,
        )
        fit_history_p = {}
        # declare tensor names outside of time_index loop
        # so we can reuse them as staring values for each successive time sample.
        fg_r = None
        fg_i = None
        g_r = None
        fg_i = None
        for time_index in range(uvdata.Ntimes):
            rmsdata = np.sqrt(
                np.mean(
                    np.abs(
                        uvdata.data_array[time_index :: uvdata.Ntimes, 0, :, polnum][
                            ~uvdata.flag_array[time_index :: uvdata.Ntimes, 0, :, polnum]
                        ]
                    )
                    ** 2.0
                )
            )
            echo(
                f"{datetime.datetime.now()} Working on time {time_index + 1} of {uvdata.Ntimes}...\n",
                verbose=verbose,
            )
            # pull data for pol out of raveled uvdata object and into dicts of 1d tf.Tensor objects for processing..
            echo(f"{datetime.datetime.now()} Tensorizing Data...\n", verbose=verbose)
            data_r, data_i, wgts = tensorize_pbl_data_dictionary(
                uvdata,
                dtype=dtype,
                time_index=time_index,
                polarization=pol,
                scale_factor=rmsdata,
                red_grps=red_grps,
                weights=weights,
            )
            echo(f"{datetime.datetime.now()} Tensorizing Gains...\n", verbose=verbose)
            gains_r, gains_i = tensorize_gains(gains, dtype=dtype, time_index=time_index, polarization=pol)
            echo(
                f"{datetime.datetime.now()} Tensorizing Foreground coeffs...\n",
                verbose=verbose,
            )
            # only generate fresh fg_r, fg_i if time_index == 0
            # otherwise, use values from previous iteration of loop.
            if time_index == 0:
                fg_r, fg_i = tensorize_fg_coeffs(
                    uvdata=sky_model,
                    model_component_dict=fg_model_comps,
                    dtype=dtype,
                    time_index=time_index,
                    polarization=pol,
                    scale_factor=rmsdata,
                )

            cal_loss = lambda g_r, g_i, fg_r, fg_i: cal_loss_dictionary(
                gains_re=g_r,
                gains_im=g_i,
                ants_map=ants_map,
                fg_coeffs_re=fg_r,
                fg_coeffs_im=fg_i,
                data_re=data_r,
                data_im=data_i,
                wgts=wgts,
                fg_range_map=fg_range_map,
                components_map=model_comps_map,
            )

            (gains_r, gains_i, fg_r, fg_i, fit_history_p[time_index],) = fit_gains_and_foregrounds(
                g_r=gains_r,
                g_i=gains_i,
                fg_r=fg_r,
                fg_i=fg_i,
                loss_function=cal_loss,
                record_var_history=record_var_history,
                record_var_history_interval=record_var_history_interval,
                optimizer=optimizer,
                use_min=use_min,
                freeze_model=freeze_model,
                notebook_progressbar=notebook_progressbar,
                verbose=verbose,
                tol=tol,
                maxsteps=maxsteps,
                **opt_kwargs,
            )

            # insert model values.
            insert_model_into_uvdata_dictionary(
                uvdata=model,
                time_index=time_index,
                polarization=pol,
                model_comps_map=model_comps_map,
                fg_coeffs_re=fg_r,
                fg_coeffs_im=fg_i,
                scale_factor=rmsdata,
                fg_range_map=fg_range_map,
            )
            insert_gains_into_uvcal(
                uvcal=gains,
                time_index=time_index,
                polarization=pol,
                gains_re=gains_r,
                gains_im=gains_i,
            )

        fit_history[polnum] = fit_history_p
        if not freeze_model:
            renormalize(
                uvdata_reference_model=sky_model,
                uvdata_deconv=model,
                gains=gains,
                polarization=pol,
                uvdata_flags=uvdata,
            )

    model_with_gains = cal_utils.apply_gains(model, gains, inverse=True)
    if not correct_model:
        model = model_with_gains
    resid.data_array -= model_with_gains.data_array
    if correct_resid:
        resid = cal_utils.apply_gains(resid, gains)

    return model, resid, gains, fit_history


def calibrate_and_model_mixed(
    uvdata,
    horizon=1.0,
    min_dly=0.0,
    offset=0.0,
    ant_dly=0.0,
    include_autos=False,
    verbose=False,
    red_tol=1.0,
    red_tol_freq=0.5,
    n_angle_bins=200,
    notebook_progressbar=False,
    use_redundancy=False,
    use_tensorflow_to_derive_modeling_comps=False,
    eigenval_cutoff=1e-10,
    dtype_matinv=np.float64,
    model_comps=None,
    **fitting_kwargs,
):
    """Simultaneously solve for gains and model foregrounds with a mix of DPSS vectors
        for baselines with no frequency redundancy and simple_cov components for
        groups of baselines that have some frequency redundancy.


    Parameters
    ----------
    uvdata: UVData object.
        dataset to calibrate and filter.
    horizon: float, optional
        fraction of baseline delay length to model with dpss modes
        unitless.
        default is 1.
    min_dly: float, optional
        minimum delay to model with dpss models.
        in units of ns.
        default is 0.
    offset: float optional
        offset off of horizon wedge to include in dpss delay range.
        in units of ns.
        default is 0.
    ant_dly: float, optional
        intrinsic chromaticity of each antenna element
        in units of ns.
        default is 0.
    include_autos: bool, optional
        if true, include autocorrelations in fitting.
        default is False.
    verbose: bool, optional
        lots of text output
        default is False.
    red_tol: float, optional
        tolerance for treating baselines as redundant (meters)
        default is 1.0
    red_tol_freq: float, optional
        tolerance for treating two baselines as having some
        frequency redundancy. When frequency redundancy exists, baselines
        will be modeled jointly.
    n_angle_bins: int, optional
        number of angular bins to use between -pi and pi to compare baselines
        default is 200
    notebook_progressbar: bool, optional
        if True, show graphical notebook progress bar that looks good in jupyter.
        default is False.
    use_redundancy: bool, optional
        If True, model all baselines within each redundant group with the same components
        If False, model each baseline within each redundant group with sepearate components.
        default is False.
    use_tensorflow_to_derive_modeling_comps: bool, optional
        Use tensorflow methods to derive multi-baseline modeling components.
        recommended if you have a GPU with enough memory to perform spectral decomposition
        of multi-baseline covariance matrices.
    eigenval_cutoff: float, optional
        threshold of eigenvectors to include in modeling components.
    dtype_matinv: numpy.dtype, optional
        data type to use for deriving modeling components.
        default is np.float64 (need higher precision for cov-mat like calculation)
    fitting_kwargs: kwarg dict
        additional kwargs for calibrate_and_model_tensor.
        see docstring of calibrate_and_model_tensor.

    Returns
    -------
    model: UVData object
        uvdata object containing DPSS model of intrinsic foregrounds.
    resid: UVData object
        uvdata object containing residuals after subtracting model times gains and applying gains.
    gains: UVCal object
        uvcal object containing fitted gains.
    fit_history:
        dictionary containing fit history for each time-step and polarization in the data with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coeffs
            'fg_i': imag part of foreground coeffs
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    # get fitting groups
    fitting_grps, blvecs, _, _ = modeling.get_uv_overlapping_grps_conjugated(
        uvdata,
        remove_redundancy=not (use_redundancy),
        red_tol=red_tol,
        include_autos=include_autos,
        red_tol_freq=red_tol_freq,
        n_angle_bins=n_angle_bins,
        notebook_progressbar=notebook_progressbar,
    )
    if model_comps is None:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        def compute_model_comps(procnum, return_dict):
            return_dict[procnum] = modeling.yield_mixed_comps(
                fitting_grps,
                blvecs,
                uvdata.freq_array[0],
                eigenval_cutoff=eigenval_cutoff,
                use_tensorflow=use_tensorflow_to_derive_modeling_comps,
                ant_dly=ant_dly,
                horizon=horizon,
                offset=offset,
                min_dly=min_dly,
                verbose=verbose,
                dtype=dtype_matinv,
                notebook_progressbar=notebook_progressbar,
            )

        proc = multiprocessing.Process(target=compute_model_comps, args=(0, return_dict))
        proc.start()
        proc.join()
        model_comps = return_dict[0]

    (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
        uvdata=uvdata,
        fg_model_comps=model_comps,
        include_autos=include_autos,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        **fitting_kwargs,
    )
    return model, resid, gains, fitted_info


def calibrate_and_model_dpss(
    uvdata,
    horizon=1.0,
    min_dly=0.0,
    offset=0.0,
    include_autos=False,
    verbose=False,
    modeling_paradigm="dictionary",
    red_tol=1.0,
    **fitting_kwargs,
):
    """Simultaneously solve for gains and model foregrounds with DPSS vectors.

    Parameters
    ----------
    uvdata: UVData object.
        dataset to calibrate and filter.
    horizon: float, optional
        fraction of baseline delay length to model with dpss modes
        unitless.
        default is 1.
    min_dly: float, optional
        minimum delay to model with dpss models.
        in units of ns.
        default is 0.
    offset: float optional
        offset off of horizon wedge to include in dpss delay range.
        in units of ns.
        default is 0.
    include_autos: bool, optional
        if true, include autocorrelations in fitting.
        default is False.
    verbose: bool, optional
        lots of text output
        default is False.
    modeling: str, optional
        specify method for modeling the data / computing loss function
        supported options are 'sparse_tensor' and 'dictionary'
        'sparse_tensor' represents each time and polarization as an
        Nants^2 x Nfreqs tensor and per-baseline modeling components
        as Nants^2 x Nfreqs x Ncomponents sparse tensor. The impact of gains
        by taking the outer product of the Nants gain arrays. This is more memory
        intensive but better optimized to take advantage of tensor operations,
        especially on GPU. 'dictionary' mode represents visibilities in a dictionary
        of spectra keyed to antenna pairs which is more memory efficient but at a loss
        of performance on GPUs.
    red_tol: float, optional
        tolerance for treating baselines as redundant (meters)
        default is 1.0

    fitting_kwargs: kwarg dict
        additional kwargs for calibrate_and_model_pbl.
        see docstring of calibrate_and_model_pbl.

    Returns
    -------
    model: UVData object
        uvdata object containing DPSS model of intrinsic foregrounds.
    resid: UVData object
        uvdata object containing residuals after subtracting model times gains and applying gains.
    gains: UVCal object
        uvcal object containing fitted gains.
    fit_history:
        dictionary containing fit history for each time-step and polarization in the data with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coeffs
            'fg_i': imag part of foreground coeffs
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    dpss_model_comps = modeling.yield_pbl_dpss_model_comps(
        uvdata,
        horizon=horizon,
        min_dly=min_dly,
        offset=offset,
        include_autos=include_autos,
        red_tol=red_tol,
    )
    if modeling_paradigm == "dictionary":
        # get rid of fitting group level for the dictionary method.
        (model, resid, gains, fitted_info,) = calibrate_and_model_pbl_dictionary_method(
            uvdata=uvdata,
            fg_model_comps=dpss_model_comps,
            include_autos=include_autos,
            verbose=verbose,
            **fitting_kwargs,
        )
    elif modeling_paradigm == "sparse_tensor":
        (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
            uvdata=uvdata,
            fg_model_comps=dpss_model_comps,
            include_autos=include_autos,
            verbose=verbose,
            **fitting_kwargs,
        )
    return model, resid, gains, fitted_info


def read_calibrate_and_model_pbl(
    infilename,
    incalfilename=None,
    refmodelname=None,
    residfilename=None,
    modelfilename=None,
    calfilename=None,
    model_basis="dpss",
    clobber=False,
    **cal_kwargs,
):
    """Driver

    Parameters
    ----------
    infilename: str
        path to the input uvh5 data file with data to calibrate and filter.
    incalefilename: str, optional
        path to input calfits calibration file to use as a starting point for gain solutions.
    refmodelname: str, optional
        path to an optional reference sky model that can be used to set initial gains and foreground coeffs.
        default is None -> initial foreground coeffs set by
    residfilename: str, optional
        path to file to output uvh5 file that stores the calibrated residual file.
        default is None -> no resid file will be written.
    modelfilename: str, optional
        path to output uvh5 file that stores the intrinsic foreground model.
        default is None -> no modelfile will be writen.
    filterfilename: str, optional
        path to output uvh5 file that stores residual plus foreground model, this is
        the fully calibrated data that includes both a calibrated foreground model
        and residual.
        default is None -> no filterfile will be written.
    calfilename: str, optional
        path to output calfits file to write gain estimates too.
    modeling basis: str, optional
        string specifying the per-baseline basis functions to use for modeling foregrounds.
        default is 'dpss'. Currently, only 'dpss' is supported.
    clobber: bool, optional
        overwrite existing output files.
        default is False.
    cal_kwargs: kwarg_dict.
        kwargs for calibrate_data_model_dpss and calibrate_and_model_pbl
        see the docstrings of these functions for more details.
    """
    # initialize uvdata
    uvdata = UVData()
    uvdata.read_uvh5(infilename)
    # initalize input calibration
    if incalfilename is not None:
        gains = UVCal()
        gains.read_calfits(incalfilename)
    else:
        gains = None
    if refmodelname is not None:
        sky_model = UVData()
        sky_model.read_uvh5(refmodelname)
    else:
        sky_model = None
    if model_basis == "dpss":
        model, resid, gains, fitted_info = calibrate_and_model_dpss(
            uvdata=uvdata, sky_model=sky_model, gains=gains, **cal_kwargs
        )
    else:
        raise NotImplementedError("only 'dpss' modeling basis is implemented.")
    if residfilename is not None:
        resid.write_uvh5(residfilename, clobber=clobber)
    if modelfilename is not None:
        model.write_uvh5(modelfilename, clobber=clobber)
    if calfilename is not None:
        gains.write_calfits(calfilename, clobber=clobber)
    return model, resid, gains, fitted_info


def red_calibrate_and_model_dpss_argparser():
    """Get argparser for calibrating and filtering.

    Parameters
    ----------
    N/A

    Returns
    -------
    ap: argparse.ArgumentParser object.
        parser for running read_calibrate_and_filter_data_pbl with model_basis='dpss'

    """
    ap = argparse.ArgumentParser(
        description="Simultaneous Gain Calibration and Filtering of Foregrounds using DPSS modes"
    )
    io_opts = ap.add_argument_group(title="I/O options.")
    io_opts.add_argument("infilename", type=str, help="Path to data file to be calibrated and modeled.")
    io_opts.add_argument(
        "--incalfilename",
        type=str,
        help="Path to optional initial gain files.",
        default=None,
    )
    io_opts.add_argument(
        "--refmodelname",
        type=str,
        help="Path to a reference sky model that can be used to initialize foreground coeffs and set overall flux scale and phase.",
    )
    io_opts.add_argument(
        "--residfilename",
        type=str,
        help="Path to write output uvh5 residual.",
        default=None,
    )
    io_opts.add_argument(
        "--modelfilename",
        type=str,
        help="Path to write output uvh5 model.",
        default=None,
    )
    io_opts.add_argument("--calfilename", type=str, help="path to write output calibration gains.")
    fg_opts = ap.add_argument_group(title="Options for foreground modeling.")
    fg_opts.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="Fraction of horizon delay to model with DPSS modes.",
    )
    fg_opts.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Offset off of horizon delay (in ns) to model foregrounds with DPSS modes.",
    )
    fg_opts.add_argument(
        "--min_dly",
        type=float,
        default=0.0,
        help="minimum delay, regardless of baseline length, to model foregrounds with DPSS modes.",
    )
    fit_opts = ap.add_argument_group(title="Options for fitting and optimization")
    fit_opts.add_argument(
        "--freeze_model",
        default=False,
        action="store_true",
        help="Only optimize gains (freeze foreground model on existing data ore user provided sky-model file).",
    )
    fit_opts.add_argument(
        "--optimizer",
        default="Adamax",
        type=str,
        help="Optimizer to use in gradient descent. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers",
    )
    fit_opts.add_argument(
        "--tol",
        type=float,
        default=1e-14,
        help="Halt optimization if loss (chsq / ndata) changes by less then this amount.",
    )
    fit_opts.add_argument(
        "--maxsteps",
        type=int,
        default=10000,
        help="Maximum number of steps to carry out optimization to",
    )
    fit_opts.add_argument("--verbose", default=False, action="store_true", help="Lots of outputs.")
    fit_opts.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        help="Initial learning rate for optimization",
    )
    return ap
