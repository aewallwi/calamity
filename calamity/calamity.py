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


OPTIMIZERS = {
    "Adadelta": tf.optimizers.Adadelta,
    "Adam": tf.optimizers.Adam,
    "Adamax": tf.optimizers.Adamax,
    "Ftrl": tf.optimizers.Ftrl,
    "Nadam": tf.optimizers.Nadam,
    "SGD": tf.optimizers.SGD,
    "RMSprop": tf.optimizers.RMSprop,
}


def tensorize_fg_model_comps_dict(
    fg_model_comps_dict,
    ants_map,
    nfreqs,
    sparse_threshold=1e-1,
    dtype=np.float32,
    notebook_progressbar=False,
    verbose=False,
    single_bls_as_sparse=True,
):
    """Convert per-baseline model components into a Ndata x Ncomponent tensor

    Parameters
    ----------
    fg_model_comps_dict: dictionary
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
    single_bls_as_sparse: bool, optional
        if True, store single baselines in a sparse tensor

    Returns
    -------
    fg_comps_chunked: list
        list of tf.Tensor objects representing each fitting group with more then a single baseline
        or tf.sparse.SparseTensor objects representing all single baseline fitting groups together.
        Shape of each tf.Tensor object is Ndata x Ncomponents (where Ndata = Nbls * Nfreqs and Nbls is number of baselines in modeling group)
        Shape of each tf.sparse.Sparse Tensor object is (Nant * Nant * Nfreq) x Ncomponents
    data_inds:
        list of float32 tf.Tensor objects representing data_indices (in 1d raveled baseline * nfreqs + freq) format
    vector_inds:
        list of float32 tf.Tensor objects representing vector indices
    """
    echo(
        f"{datetime.datetime.now()} Computing foreground components matrices...\n",
        verbose=verbose,
    )

    # do a forward pass and determine dense shape of the sparse matrix
    sparse_number_of_elements = 0
    nvectors = 0
    nants_data = len(ants_map)
    echo("Determining number of non-zero elements", verbose=verbose)
    for modeling_grp in PBARS[notebook_progressbar](fg_model_comps_dict):
        if len(modeling_grp) == 1:
            for vind in range(fg_model_comps_dict[modeling_grp].shape[1]):
                for grpnum, red_grp in enumerate(modeling_grp):
                    for ap in red_grp:
                        sparse_number_of_elements += nfreqs
                nvectors += 1

    dense_shape = (int(nants_data ** 2.0 * nfreqs), nvectors)
    dense_number_of_elements = dense_shape[0] * dense_shape[1]

    sparseness = sparse_number_of_elements / dense_number_of_elements
    # build maps between i, j correlation indices and single baselines
    # so we can build up sparse matrix representation of non-overlapping baselines.
    echo(f"Fraction of modeling matrix with nonzero values is {(sparseness):.4e}", verbose=verbose)
    echo(f"Generating map between i,j indices and foreground modeling keys", verbose=verbose)
    modeling_grps = {}
    red_grp_nums = {}
    start_inds = {}
    stop_inds = {}
    start_ind = 0
    for modeling_grp in fg_model_comps_dict:
        if len(modeling_grp) == 1:
            stop_ind = start_ind + fg_model_comps_dict[modeling_grp].shape[1]
            for red_grp_num, red_grp in enumerate(modeling_grp):
                for ap in red_grp:
                    i, j = ants_map[ap[0]], ants_map[ap[1]]
                    modeling_grps[(i, j)] = modeling_grp
                    red_grp_nums[(i, j)] = red_grp_num
                    start_inds[(i, j)] = start_ind
                    stop_inds[(i, j)] = stop_ind

            start_ind = stop_ind
    ordered_ijs = sorted(list(modeling_grps.keys()))

    comp_inds = np.zeros((sparse_number_of_elements, 2), dtype=np.int64)
    comp_vals = np.zeros(sparse_number_of_elements, dtype=dtype)

    echo(f"{datetime.datetime.now()} Filling out modeling vectors for non-overlapping baselines...\n", verbose=verbose)

    if single_bls_as_sparse:
        spinds = None
        for i, j in PBARS[notebook_progressbar](ordered_ijs):
            blind = i * nants_data + j
            grpnum = red_grp_nums[(i, j)]
            fitgrp = modeling_grps[(i, j)]
            start_ind = start_inds[(i, j)]
            stop_ind = stop_inds[(i, j)]
            nvecs = stop_ind - start_ind
            dinds = np.hstack(
                [np.ones(nvecs) * dind for dind in np.arange(blind * nfreqs, (blind + 1) * nfreqs)]
            ).astype(np.int64)
            matcols = np.hstack([np.arange(start_ind, stop_ind) for i in np.arange(nfreqs)]).astype(np.int64)
            bl_mvec = fg_model_comps_dict[fitgrp][grpnum * nfreqs : (grpnum + 1) * nfreqs].astype(dtype).flatten()
            ninds = len(dinds)
            if spinds is None:
                spinds = np.arange(ninds).astype(np.int64)
            else:
                spinds = np.arange(ninds).astype(np.int64) + spinds[-1] + 1
            comp_vals[spinds] = bl_mvec
            comp_inds[spinds, 0], comp_inds[spinds, 1] = dinds, matcols

        fg_comps_sparse = tf.sparse.SparseTensor(indices=comp_inds, values=comp_vals, dense_shape=dense_shape)
        data_inds_sparse = tf.convert_to_tensor(comp_inds[:, 0], dtype=np.int64)
    else:
        fg_comps_sparse = None

    fg_comps_chunked = []
    corr_inds_chunked = []
    # now go through the fitting_groups that are not length-1
    for modeling_grp in fg_model_comps_dict:
        if len(modeling_grp) > 1 or not single_bls_as_sparse:
            # need to duplicate rows for redundant baselines.
            fg_comp_chunk = []
            corrinds = []
            # calculate data indices
            for grpnum, red_grp in enumerate(modeling_grp):
                for ap in red_grp:
                    i, j = ants_map[ap[0]], ants_map[ap[1]]
                    corrinds.append([i, j])
                    fg_comp_chunk.append(fg_model_comps_dict[modeling_grp][grpnum * nfreqs : (grpnum + 1) * nfreqs])
            fg_comp_chunk = np.vstack(fg_comp_chunk)
            fg_comps_chunked.append(tf.convert_to_tensor(fg_comp_chunk, dtype=dtype))
            corr_inds_chunked.append(corrinds)
    if len(fg_comps_chunked) == 0:
        fg_comps_chunked = None
        corr_inds_chunked = None

    return fg_comps_sparse, fg_comps_chunked, corr_inds_chunked


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


def yield_fg_model_array(
    nants,
    nfreqs,
    fg_comps_chunked=None,
    fg_coeffs_chunked=None,
    corr_inds_chunked=None,
    fg_comps_sparse=None,
    fg_coeffs_sparse=None,
):
    """Compute sparse tensor foreground model.

    Parameters
    ----------
    data_inds: tf.ragged.RaggedTensor



    fg_comps: tf.ragged.RaggedTensor

    fg_coeffs: tf.ragged.RaggedTensor

    sparse_comps: tf.sparse.SparseTensor or tf.Tensor object
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
    model = None
    if fg_comps_sparse is not None:
        model = tf.reshape(
            tf.sparse.sparse_dense_matmul(fg_comps_sparse, fg_coeffs_sparse), (nants, nants, nfreqs)
        ).numpy()
    else:
        model = np.zeros((nants, nants, nfreqs))
    if fg_comps_chunked is not None:
        ngrps = len(fg_comps_chunked)
        for gnum in tf.range(ngrps):
            gchunk = tf.reduce_sum(fg_comps_chunked[gnum] * fg_coeffs_chunked[gnum], axis=1).numpy()
            for rnum, (i, j) in enumerate(corr_inds_chunked[gnum]):
                model[i, j] = gchunk[rnum * nfreqs : (rnum + 1) * nfreqs]
    return model


def fit_gains_and_foregrounds(
    g_r,
    g_i,
    fg_r_chunked=None,
    fg_i_chunked=None,
    fg_r_sparse=None,
    fg_i_sparse=None,
    data_r=None,
    data_i=None,
    wgts=None,
    fg_comps_sparse=None,
    fg_comps_chunked=None,
    corr_inds_chunked=None,
    loss_function=None,
    use_min=False,
    tol=1e-14,
    maxsteps=10000,
    optimizer="Adamax",
    freeze_model=False,
    verbose=False,
    notebook_progressbar=False,
    dtype=np.float32,
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
    """

    # initialize the optimizer.
    opt = OPTIMIZERS[optimizer](**opt_kwargs)
    # set up history recording
    fit_history = {"loss": []}

    min_loss = 9e99
    echo(
        f"{datetime.datetime.now()} Building Computational Graph...\n",
        verbose=verbose,
    )
    nants = g_r.shape[0]
    nfreqs = g_r.shape[1]
    # copy data into ragged tensors for each fitting group if necessary.
    if fg_comps_chunked is not None:
        ngrps = len(fg_comps_chunked)
        data_r_chunked = [tf.reshape(tf.gather_nd(data_r, corr_inds_chunked[gnum]), [-1]) for gnum in range(ngrps)]
        data_i_chunked = [tf.reshape(tf.gather_nd(data_i, corr_inds_chunked[gnum]), [-1]) for gnum in range(ngrps)]
        ant0_inds = [list(np.asarray(corr_inds_chunked[gnum])[:, 0]) for gnum in range(ngrps)]
        ant1_inds = [list(np.asarray(corr_inds_chunked[gnum])[:, 1]) for gnum in range(ngrps)]
        wgts_chunked = [tf.reshape(tf.gather_nd(wgts, corr_inds_chunked[gnum]), [-1]) for gnum in range(ngrps)]

    g_r = tf.Variable(g_r)
    g_i = tf.Variable(g_i)
    if not freeze_model:
        if fg_comps_sparse is not None and fg_comps_chunked is None:
            fg_r_sparse = tf.Variable(fg_r_sparse)
            fg_i_sparse = tf.Variable(fg_i_sparse)
            vars = [g_r, g_i, fg_r_sparse, fg_i_sparse]
        elif fg_comps_sparse is None and fg_comps_chunked is not None:
            fg_r_chunked = [tf.Variable(fgr) for fgr in fg_r_chunked]
            fg_i_chunked = [tf.Variable(fgi) for fgi in fg_i_chunked]
            vars = [g_r, g_i] + fg_r_chunked + fg_i_chunked
        else:
            fg_r_sparse = tf.Variable(fg_r_sparse)
            fg_i_sparse = tf.Variable(fg_i_sparse)
            fg_r_chunked = [tf.Variable(fgr) for fgr in fg_r_chunked]
            fg_i_chunked = [tf.Variable(fgi) for fgi in fg_i_chunked]
            vars = [g_r, g_i, fg_r_sparse, fg_i_sparse] + fg_r_chunked + fg_i_chunked
    else:
        vars = [g_r, g_i]

    if fg_comps_sparse is not None:

        def loss_function_sparse():
            # start with sparse components.
            grgr = tf.einsum("ik,jk->ijk", g_r, g_r)
            gigi = tf.einsum("ik,jk->ijk", g_i, g_i)
            grgi = tf.einsum("ik,jk->ijk", g_r, g_i)
            gigr = tf.einsum("ik,jk->ijk", g_i, g_r)
            vr = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps_sparse, fg_r_sparse), (nants, nants, nfreqs))
            vi = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps_sparse, fg_i_sparse), (nants, nants, nfreqs))
            model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
            model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
            cal_loss = tf.reduce_sum((tf.square(data_r - model_r) + tf.square(data_i - model_i)) * wgts)
            return cal_loss

    else:

        def loss_function_sparse():
            return tf.constant(0.0, dtype=dtype)

    if fg_comps_chunked is not None:

        def loss_function_chunked():
            cal_loss = tf.constant(0.0, dtype=dtype)
            # now deal with dense components
            for gnum in tf.range(ngrps):
                gr0 = tf.gather(g_r, ant0_inds[gnum])
                gr1 = tf.gather(g_r, ant1_inds[gnum])
                gi0 = tf.gather(g_i, ant0_inds[gnum])
                gi1 = tf.gather(g_i, ant1_inds[gnum])
                grgr = tf.reshape(gr0 * gr1, [-1])
                gigi = tf.reshape(gi0 * gi1, [-1])
                grgi = tf.reshape(gr0 * gi1, [-1])
                gigr = tf.reshape(gi0 * gr1, [-1])
                vr = tf.reduce_sum(fg_comps_chunked[gnum] * fg_r_chunked[gnum], axis=1)
                vi = tf.reduce_sum(fg_comps_chunked[gnum] * fg_i_chunked[gnum], axis=1)
                model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
                model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
                cal_loss += tf.reduce_sum(
                    (tf.square(data_r_chunked[gnum] - model_r) + tf.square(data_i_chunked[gnum] - model_i))
                    * wgts_chunked[gnum]
                )
            return cal_loss

    else:

        def loss_function_chunked():
            return tf.constant(0.0, dtype=dtype)

    def loss_function():
        return loss_function_sparse() + loss_function_chunked()

    loss_i = loss_function().numpy()
    echo(
        f"{datetime.datetime.now()} Performing Gradient Descent. Initial MSE of {loss_i:.2e}...\n",
        verbose=verbose,
    )

    for step in PBARS[notebook_progressbar](range(maxsteps)):
        with tf.GradientTape() as tape:
            loss = loss_function()
        grads = tape.gradient(loss, vars)
        opt.apply_gradients(zip(grads, vars))
        fit_history["loss"].append(loss.numpy())
        if use_min and fit_history["loss"][-1] < min_loss:
            # store the g_r, g_i, fg_r, fg_i values that minimize loss
            # in case of overshoot.
            min_loss = fit_history["loss"][-1]
            g_r_opt = g_r.value()
            g_i_opt = g_i.value()
            if not freeze_model:
                if fg_comps_chunked is not None:
                    fg_r_chunked_opt = [fgr.value() for fgr in fg_r_chunked]
                    fg_i_chunked_opt = [fgi.value() for fgi in fg_i_chunked]
                if fg_comps_sparse is not None:
                    fg_r_sparse_opt = fg_r_sparse.value()
                    fg_i_sparse_opt = fg_i_sparse.value()

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
            if fg_comps_chunked is not None:
                fg_r_chunked_opt = [fgr.value() for fgr in fg_r_chunked]
                fg_i_chunked_opt = [fgi.value() for fgi in fg_i_chunked]
            if fg_comps_sparse is not None:
                fg_r_sparse_opt = fg_r_sparse.value()
                fg_i_sparse_opt = fg_i_sparse.value()
        else:
            if fg_comps_chunked is not None:
                fg_r_chunked_opt = fg_r_chunked
                fg_i_chunked_opt = fg_i_chunked
            if fg_comps_sparse is not None:
                fg_r_sparse_opt = fg_r_sparse
                fg_i_sparse_opt = fg_i_sparse

    if fg_comps_sparse is None:
        fg_r_sparse_opt = None
        fg_i_sparse_opt = None
    if fg_comps_chunked is None:
        fg_r_chunked_opt = None
        fg_i_chunked_opt = None

    echo(
        f"{datetime.datetime.now()} Finished Gradient Descent. MSE of {min_loss:.2e}...\n",
        verbose=verbose,
    )
    return g_r_opt, g_i_opt, fg_r_chunked_opt, fg_i_chunked_opt, fg_r_sparse_opt, fg_i_sparse_opt, fit_history


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
    model_r: np.ndarray
        an Nants_data x Nants_data x Nfreqs np.ndarray with real parts of data
    model_i: np.ndarray
        an Nants_data x Nants_data x Nfreqs np.ndarray with imag parts of model
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
                model = model_r[i, j] + 1j * model_i[i, j]
            else:
                dinds = uvdata.antpair2ind(ap[::-1])[time_index]
                model = model_r[i, j] - 1j * model_i[i, j]
            uvdata.data_array[dinds, 0, :, polnum] = model * scale_factor


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


def tensorize_fg_coeffs(
    uvdata,
    fg_model_comps_dict,
    time_index,
    polarization,
    scale_factor=1.0,
    dtype=np.float32,
    single_bls_as_sparse=True,
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

    fg_coeffs_re_chunked = []
    fg_coeffs_im_chunked = []
    fg_coeffs_re_sparse = []
    fg_coeffs_im_sparse = []
    # first get non-sparse components
    for fit_grp in fg_model_comps_dict:
        fg_coeffs = 0.0
        for grpnum, red_grp in enumerate(fit_grp):
            for ap in red_grp:
                bl = ap + (polarization,)
                fg_coeffs += (
                    uvdata.get_data(bl)[time_index] * ~uvdata.get_flags(bl)[time_index]
                ) @ fg_model_comps_dict[fit_grp][grpnum * uvdata.Nfreqs : (grpnum + 1) * uvdata.Nfreqs]

        if len(fit_grp) == 1 and single_bls_as_sparse:
            fg_coeffs_re_sparse.extend(list(fg_coeffs.real / scale_factor))
            fg_coeffs_im_sparse.extend(list(fg_coeffs.imag / scale_factor))
        else:
            fg_coeffs_re_chunked.append(tf.convert_to_tensor(fg_coeffs.real / scale_factor, dtype=dtype))
            fg_coeffs_im_chunked.append(tf.convert_to_tensor(fg_coeffs.imag / scale_factor, dtype=dtype))

    fg_coeffs_re_sparse = tf.convert_to_tensor(fg_coeffs_re_sparse, dtype=dtype)
    fg_coeffs_re_sparse = tf.reshape(fg_coeffs_re_sparse, (fg_coeffs_re_sparse.shape[0], 1))
    fg_coeffs_im_sparse = tf.convert_to_tensor(fg_coeffs_im_sparse, dtype=dtype)
    fg_coeffs_im_sparse = tf.reshape(fg_coeffs_im_sparse, (fg_coeffs_im_sparse.shape[0], 1))
    return fg_coeffs_re_sparse, fg_coeffs_im_sparse, fg_coeffs_re_chunked, fg_coeffs_im_chunked


def calibrate_and_model_tensor(
    uvdata,
    fg_model_comps_dict,
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
    use_redundancy=False,
    notebook_progressbar=False,
    correct_resid=False,
    correct_model=False,
    weights=None,
    sparse_threshold=1e-1,
    single_bls_as_sparse=True,
    **opt_kwargs,
):
    """Perform simultaneous calibration and foreground fitting using sparse tensors.


    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    fg_model_comps_dict: dictionary
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
    use_sparse: bool, optional
        use sparse representation if True.
        default is None -> use sparse_threshold to determine whether to use sparse representation.
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
    """
    antpairs_data = uvdata.get_antpairs()
    if not include_autos:
        antpairs_data = set([ap for ap in antpairs_data if ap[0] != ap[1]])
    uvdata = uvdata.select(inplace=False, bls=[ap for ap in antpairs_data])

    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)

    # get redundant groups
    red_grps = []
    for fit_grp in fg_model_comps_dict.keys():
        for red_grp in fit_grp:
            red_grps.append(red_grp)

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
    fg_comps_sparse, fg_comps_chunked, corr_inds_chunked = tensorize_fg_model_comps_dict(
        fg_model_comps_dict=fg_model_comps_dict,
        ants_map=ants_map,
        dtype=dtype,
        nfreqs=sky_model.Nfreqs,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        sparse_threshold=sparse_threshold,
        single_bls_as_sparse=single_bls_as_sparse,
    )
    echo(
        f"{datetime.datetime.now()}Finished Converting Foreground Modeling Components to Tensors...\n",
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
            fg_r_sparse, fg_i_sparse, fg_r_chunked, fg_i_chunked = tensorize_fg_coeffs(
                uvdata=sky_model,
                fg_model_comps_dict=fg_model_comps_dict,
                dtype=dtype,
                time_index=time_index,
                polarization=pol,
                scale_factor=rmsdata,
                single_bls_as_sparse=single_bls_as_sparse,
            )

            (
                gains_r,
                gains_i,
                fg_r_chunked,
                fg_i_chunked,
                fg_r_sparse,
                fg_i_sparse,
                fit_history_p[time_index],
            ) = fit_gains_and_foregrounds(
                g_r=gains_r,
                g_i=gains_i,
                fg_r_chunked=fg_r_chunked,
                fg_i_chunked=fg_i_chunked,
                fg_r_sparse=fg_r_sparse,
                fg_i_sparse=fg_i_sparse,
                data_r=data_r,
                data_i=data_i,
                wgts=wgts,
                fg_comps_sparse=fg_comps_sparse,
                fg_comps_chunked=fg_comps_chunked,
                corr_inds_chunked=corr_inds_chunked,
                optimizer=optimizer,
                use_min=use_min,
                freeze_model=freeze_model,
                notebook_progressbar=notebook_progressbar,
                verbose=verbose,
                tol=tol,
                dtype=dtype,
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
                model_r=yield_fg_model_array(
                    fg_comps_sparse=fg_comps_sparse,
                    fg_comps_chunked=fg_comps_chunked,
                    fg_coeffs_chunked=fg_r_chunked,
                    fg_coeffs_sparse=fg_r_sparse,
                    corr_inds_chunked=corr_inds_chunked,
                    nants=uvdata.Nants_data,
                    nfreqs=uvdata.Nfreqs,
                ),
                model_i=yield_fg_model_array(
                    fg_comps_sparse=fg_comps_sparse,
                    fg_comps_chunked=fg_comps_chunked,
                    fg_coeffs_chunked=fg_i_chunked,
                    fg_coeffs_sparse=fg_i_sparse,
                    corr_inds_chunked=corr_inds_chunked,
                    nants=uvdata.Nants_data,
                    nfreqs=uvdata.Nfreqs,
                ),
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
    require_exact_angle_match=True,
    angle_match_tol=1e-3,
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
        require_exact_angle_match=require_exact_angle_match,
        angle_match_tol=angle_match_tol,
    )

    model_comps_dict = modeling.yield_mixed_comps(
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

    (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
        uvdata=uvdata,
        fg_model_comps_dict=model_comps_dict,
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
    notebook_progressbar=False,
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
    """
    dpss_model_comps_dict = modeling.yield_pbl_dpss_model_comps(
        uvdata,
        horizon=horizon,
        min_dly=min_dly,
        offset=offset,
        include_autos=include_autos,
        red_tol=red_tol,
        notebook_progressbar=notebook_progressbar,
    )

    (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
        uvdata=uvdata,
        fg_model_comps_dict=dpss_model_comps_dict,
        include_autos=include_autos,
        verbose=verbose,
        **fitting_kwargs,
    )

    return model, resid, gains, fitted_info
