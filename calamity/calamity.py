import numpy as np
import tensorflow as tf
from pyuvdata import UVData, UVCal
from . import utils
import copy
import argparse
import itertools
from .utils import echo
import datetime
from pyuvdata import utils as uvutils

OPTIMIZERS = {
    "Adadelta": tf.optimizers.Adadelta,
    "Adam": tf.optimizers.Adam,
    "Adamax": tf.optimizers.Adamax,
    "Ftrl": tf.optimizers.Ftrl,
    "Nadam": tf.optimizers.Nadam,
    "SGD": tf.optimizers.SGD,
    "RMSprop": tf.optimizers.RMSprop,
}


def sparse_tensorize_pbl_fg_model_comps(red_grps, fg_model_comps, ants_map, dtype=np.float32):
    """Convert per-baseline model components into a sparse Ndata x Ncomponent tensor

    Parameters
    ----------
    red_grps: list of lists of int 2-tuples
        a list of lists of 2-tuples where all antenna pairs within each sublist
        are redundant with eachother. Assumes that conjugates are correctly taken.
    fg_model_comps: dict of 2-tuples as keys and numpy.ndarray as values.
        dictionary mapping int antenna-pair 2-tuples to
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    dtype: numpy.dtype
        data-type to store in sparse tensor.
        default is np.float32

    Returns
    -------
    sparse_fg_model_mat: tf.sparse.SparseTensor object
        sparse tensor object holding foreground modeling vectors with a dense shape of
        Nants^2 x Nfreqs x Nfg_comps ~ Nbls^4 x Nfreqs
    """
    sparse_number_of_elements = 0.0
    comp_inds = []
    comp_vals = []
    fgvind = 0
    nants_data = len(ants_map)
    nfreqs = len(fg_model_comps[red_grps[0][0]])
    for red_grp in red_grps:
        # calculate one Ndata x Ncomponent sparse vector per redundant baseline group.
        for vind in range(fg_model_comps[red_grp[0]].shape[1]):
            for ap in red_grp:
                i, j = ants_map[ap[0]], ants_map[ap[1]]
                blind = i * nants_data + j
                for f in range(nfreqs):
                    comp_inds.append([blind * nfreqs + f, fgvind])
                    comp_vals.append(fg_model_comps[ap][f, vind].astype(dtype))
                    sparse_number_of_elements += 1

            fgvind += 1

    # sort by comp_inds.
    sort_order = sorted(range(len(comp_inds)), key=comp_inds.__getitem__)
    comp_vals = [comp_vals[ind] for ind in sort_order]
    comp_inds = [comp_inds[ind] for ind in sort_order]
    dense_shape = (int(nants_data ** 2.0 * nfreqs), fgvind)
    sparse_fg_model_mat = tf.sparse.SparseTensor(indices=comp_inds, values=comp_vals, dense_shape=dense_shape)
    return sparse_fg_model_mat


def tensorize_data(
    uvdata,
    red_grps,
    ants_map,
    polarization,
    time_index,
    data_scale_factor=1.0,
    wgts_scale_factor=1.0,
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
    wgts_scale_factor: float, optional
        overall scaling factor to divide tensorized weights by.
        default is 1.0 (but we recommend sum of weights * flags)
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
        scaled by wgts_scale_factor
    """
    dshape = (uvdata.Nants_data, uvdata.Nants_data, uvdata.Nfreqs)
    data_r = np.zeros(dshape, dtype=dtype)
    data_i = np.zeros_like(data_r)
    wgts = np.zeros_like(data_r)
    for red_grp in red_grps:
        for ap in red_grp:
            bl = ap + (polarization,)
            data = (uvdata.get_data(bl) / data_scale_factor).astype(dtype)
            iflags = (~uvdata.get_flags(bl)).astype(dtype)
            nsamples = (uvdata.get_nsamples(bl) / wgts_scale_factor).astype(dtype)
            i, j = ants_map[ap[0]], ants_map[ap[1]]
            data_r[i, j] = data.real
            data_i[i, j] = data.imag
            wgts[i, j] = iflags * nsamples
    data_r = tf.convert_to_tensor(data_r, dtype=dtype)
    data_i = tf.convert_to_tensor(data_i, dtype=dtype)
    wgts = tf.convert_to_tensor(wgts, dtype=dtype)
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


def tensorize_pbl_model_comps_dictionary(red_grps, fg_model_comps, dtype=np.float32):
    """Helper function generating mappings for per-baseline foreground modeling.

    Generates mappings between antenna pairs and foreground basis vectors accounting for redundancies.

    Parameters
    ----------
    red_grps: list of lists of int 2-tuples
        a list of lists of 2-tuples where all antenna pairs within each sublist
        are redundant with eachother. Assumes that conjugates are correctly taken.
    fg_model_comps: dict of 2-tuples as keys and numpy.ndarray as values.
        dictionary mapping int antenna-pair 2-tuples to

    Returns
    -------
    fg_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.
    model_component_tensor_map: diction with 2-tuples as keys an tf.Tensor objects as values.
        dictionary mapping antenna pairs to
    """
    model_component_tensor_map = {}
    fg_range_map = {}
    startind = 0
    for grpnum, red_grp in enumerate(red_grps):
        # set fg_range_map value based on first antenna-pair in redundant group.
        ncomponents = fg_model_comps[red_grp[0]].shape[1]
        fg_range_map[red_grp[0]] = (startind, startind + ncomponents)
        startind += ncomponents
        for ap in red_grp:
            model_component_tensor_map[ap] = tf.convert_to_tensor(fg_model_comps[ap], dtype=dtype)
            fg_range_map[ap] = fg_range_map[red_grp[0]]
    return fg_range_map, model_component_tensor_map


def yield_fg_pbl_model_sparse_tensor(fg_comps, fg_coeffs, nants, nfreqs):
    model = tf.reshape(tf.sparse.sparse_dense_matmul(fg_comps, fg_coeffs), (nants, nants, nfreqs))
    return model


def yield_data_model_sparse_tensor(g_r, g_i, fg_comps, fg_r, fg_i, nants, nfreqs):
    grgr = tf.einsum("ik,jk->ijk", g_r, g_r)
    gigi = tf.einsum("ik,jk->ijk", g_i, g_i)
    grgi = tf.einsum("ik,jk->ijk", g_r, g_i)
    gigr = tf.einsum("ik,jk->ijk", g_i, g_r)
    vr = yield_fg_pbl_model_sparse_tensor(fg_r, nants, nfreqs)
    vi = yield_fg_pbl_model_sparse_tensor(fg_i, nants, nfreqs)
    model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
    model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
    return model_r, model_i


def cal_loss_sparse_tensor(data_r, data_i, wgts, g_r, g_i, fg_model_comps, fg_r, fg_i, nants, nfreqs):
    model_r, model_i = yield_data_model_sparse_tensor(g_r, g_i, fg_model_comps, fg_r, fg_i, nants, nfreqs)
    return tf.reduce_sum(tf.square(data_r - model_r) * wgts + tf.square(data_i - model_i) * wgts)


def fit_gains_and_foregrounds(
    g_r,
    g_i,
    fg_r,
    fg_i,
    loss_function,
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
    if notebook_progressbar:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    min_loss = 9e99
    echo(
        f"{datetime.datetime.now()} Building Computational Graph...\n",
        verbose=verbose,
    )

    @tf.function
    def cal_loss():
        return loss_function(g_r=g_r, g_i=g_i, fg_i=fg_i, fg_r=fg_r)

    loss_i = cal_loss().numpy()
    echo(
        f"{datetime.datetime.now()} Performing Gradient Descent. Initial MSE of {loss_i:.2e}...\n",
        verbose=verbose,
    )
    for step in tqdm(range(maxsteps)):
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
    polnum = np.where(uvdata.polarization_array == uvutils.polstr2num(polarization))[0][0]
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
    wgts_scale_factor=1.0,
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
    wgts_scale_factor: float, optional
        overall scaling factor to divide tensorized weights by.
        default is 1.0 (but we recommend sum of weights * flags)
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
    for red_grp in red_grps:
        for ap in red_grp:
            bl = ap + (polarization,)
            data_re[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].real / scale_factor, dtype=dtype)
            data_im[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].imag / scale_factor, dtype=dtype)
            wgts[ap] = tf.convert_to_tensor(
                ~uvdata.get_flags(bl)[time_index] * uvdata.get_nsamples(bl)[time_index] / wgts_scale_factor,
                dtype=dtype,
            )
    return data_re, data_im, wgts


def tensorize_fg_coeffs(
    uvdata,
    model_component_dict,
    red_grps,
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
    model_component_dict: dict with int 2-tuple keys and numpy.ndarray values.
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
    for red_grp in red_grps:
        ap = red_grp[0]
        bl = ap + (polarization,)
        fg_coeffs_re.extend(
            uvdata.get_data(bl).real[time_index] * ~uvdata.get_flags(bl)[time_index] @ model_component_dict[ap]
        )
        fg_coeffs_im.extend(
            uvdata.get_data(bl).imag[time_index] * ~uvdata.get_flags(bl)[time_index] @ model_component_dict[ap]
        )
    fg_coeffs_re = np.asarray(fg_coeffs_re) / scale_factor
    fg_coeffs_im = np.asarray(fg_coeffs_im) / scale_factor
    if force2d:
        fg_coeffs_re = fg_coeffs_re.reshape((len(fg_coeffs_re), 1))
        fg_coeffs_im = fg_coeffs_im.reshape((len(fg_coeffs_im), 1))
    fg_coeffs_re = tf.convert_to_tensor(fg_coeffs_re, dtype=dtype)
    fg_coeffs_im = tf.convert_to_tensor(fg_coeffs_im, dtype=dtype)

    return fg_coeffs_re, fg_coeffs_im


def calibrate_and_model_pbl_sparse_method(
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

    uvdata = uvdata.select(inplace=False, bls=[ap for ap in antpairs_data])
    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)
    filtered = copy.deepcopy(uvdata)
    if gains is None:
        echo(
            f"{datetime.datetime.now()} Gains are None. Initializing gains starting with unity...\n",
            verbose=verbose,
        )
        gains = utils.blank_uvcal_from_uvdata(uvdata)
    # if sky-model is None, initialize it to be the
    # data divided by the initial gain estimates.

    antpairs, red_grps, antpair_red_indices, _ = utils.get_redundant_groups_conjugated(
        uvdata, remove_redundancy=not (use_redundancy), include_autos=include_autos
    )

    if sky_model is None:
        echo(
            f"{datetime.datetime.now()} Sky model is None. Initializing from data...\n",
            verbose=verbose,
        )
        sky_model = utils.apply_gains(uvdata, gains)

    fit_history = {}
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    # generate sparse tensor to hold foreground components.
    echo(
        f"{datetime.datetime.now()} Computing sparse foreground components matrix...\n",
        verbose=verbose,
    )
    fg_comp_tensor = sparse_tensorize_pbl_fg_model_comps(
        red_grps=red_grps, fg_model_comps=fg_model_comps, ants_map=ants_map, dtype=dtype
    )
    echo(
        f"{datetime.datetime.now()}Finished Computing sparse foreground components matrix...\n",
        verbose=verbose,
    )
    dense_number_of_elements = np.prod(fg_comp_tensor.dense_shape.numpy())
    sparse_number_of_elements = len(fg_comp_tensor.values)
    echo(
        f"Fraction of sparse matrix with nonzero values is {(sparse_number_of_elements / dense_number_of_elements):.4e}"
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
            wgtsum = np.sum(
                uvdata.nsample_array[time_index :: uvdata.Ntimes, 0, :, polnum]
                * ~uvdata.flag_array[time_index :: uvdata.Ntimes, 0, :, polnum]
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
                wgts_scale_factor=wgtsum,
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
                red_grps=red_grps,
                model_component_dict=fg_model_comps,
                dtype=dtype,
                time_index=time_index,
                polarization=pol,
                scale_factor=rmsdata,
            )

            cal_loss = lambda g_r, g_i, fg_r, fg_i: cal_loss_sparse_tensor(
                data_r=data_r,
                gata_i=data_i,
                wgts=wgts,
                g_r=g_r,
                g_i=g_i,
                fg_model_comps=fg_comp_tensor,
            )
            # derive optimal gains and foregrounds
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
                **opt_kwargs,
            )
            # insert into model uvdata.
            insert_model_into_uvdata_tensor(
                uvdata=model,
                time_index=time_index,
                polarization=pol,
                ants_map=ants_map,
                red_grps=red_grps,
                model_r=yield_fg_pbl_model_sparse_tensor(fg_comp_tensor, fg_r, uvdata.Nants_data, uvdata.Nfreqs),
                model_i=yield_fg_pbl_model_sparse_tensor(fg_comp_tensor, fg_i, uvdata.Nants_data, uvdata.Nfreqs),
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
    model_with_gains = utils.apply_gains(model, gains, inverse=True)
    resid.data_array -= model_with_gains.data_array
    resid = utils.apply_gains(resid, gains)
    filtered = copy.deepcopy(model)
    filtered.data_array += resid.data_array

    return model, resid, filtered, gains, fit_history


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
    **opt_kwargs,
):
    """Perform simultaneous calibration and fitting of foregrounds using method that loops over baselines.

    This approach gives up on trying to invert the wedge but can be used on practically any array.
    Use this in place of calibrate_and_model_pbl_sparse_method when working on CPUs but
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

    uvdata = uvdata.select(inplace=False, bls=[ap for ap in antpairs_data])
    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)
    filtered = copy.deepcopy(uvdata)
    if gains is None:
        echo(
            f"{datetime.datetime.now()} Gains are None. Initializing gains starting with unity...\n",
            verbose=verbose,
        )
        gains = utils.blank_uvcal_from_uvdata(uvdata)
    # if sky-model is None, initialize it to be the
    # data divided by the initial gain estimates.

    antpairs, red_grps, antpair_red_indices, _ = utils.get_redundant_groups_conjugated(
        uvdata, remove_redundancy=not (use_redundancy), include_autos=include_autos
    )

    if sky_model is None:
        echo(
            f"{datetime.datetime.now()} Sky model is None. Initializing from data...\n",
            verbose=verbose,
        )
        sky_model = utils.apply_gains(uvdata, gains)

    fit_history = {}
    echo(
        f"{datetime.datetime.now()} Generating map between antenna pairs and modeling vectors...\n",
        verbose=verbose,
    )
    (
        fg_range_map,
        model_comps_map,
    ) = tensorize_pbl_model_comps_dictionary(red_grps, fg_model_comps, dtype=dtype)
    # index antenna numbers (in gain vectors).
    ants_map = {gains.antenna_numbers[i]: i for i in range(len(gains.antenna_numbers))}
    # We do fitting per time and per polarization and time.
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
            wgtsum = np.sum(
                uvdata.nsample_array[time_index :: uvdata.Ntimes, 0, :, polnum]
                * ~uvdata.flag_array[time_index :: uvdata.Ntimes, 0, :, polnum]
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
                wgts_scale_factor=wgtsum,
                red_grps=red_grps,
            )
            echo(f"{datetime.datetime.now()} Tensorizing Gains...\n", verbose=verbose)
            gains_r, gains_i = tensorize_gains(gains, dtype=dtype, time_index=time_index, polarization=pol)
            echo(
                f"{datetime.datetime.now()} Tensorizing Foreground coeffs...\n",
                verbose=verbose,
            )
            fg_r, fg_i = tensorize_fg_coeffs(
                uvdata=sky_model,
                red_grps=red_grps,
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

    model_with_gains = utils.apply_gains(model, gains, inverse=True)
    resid.data_array -= model_with_gains.data_array
    resid = utils.apply_gains(resid, gains)
    filtered = copy.deepcopy(model)
    filtered.data_array += resid.data_array

    return model, resid, filtered, gains, fit_history


def calibrate_and_model_dpss(
    uvdata,
    horizon=1.0,
    min_dly=0.0,
    offset=0.0,
    include_autos=False,
    verbose=False,
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
    fitting_kwargs: kwarg dict
        additional kwargs for calibrate_and_model_pbl.
        see docstring of calibrate_and_model_pbl.

    Returns
    -------
    model: UVData object
        uvdata object containing DPSS model of intrinsic foregrounds.
    resid: UVData object
        uvdata object containing residuals after subtracting model times gains and applying gains.
    filtered: UVData object
        uvdata object containing the sum of the residuals and instrinsic foreground model.
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
    dpss_model_comps = utils.yield_dpss_model_comps(
        uvdata,
        horizon=horizon,
        min_dly=min_dly,
        offset=offset,
        include_autos=include_autos,
    )
    (model, resid, filtered, gains, fitted_info,) = calibrate_and_model_pbl_dictionary_method(
        uvdata=uvdata,
        fg_model_comps=dpss_model_comps,
        include_autos=include_autos,
        verbose=verbose,
        **fitting_kwargs,
    )
    return model, resid, filtered, gains, fitted_info


def read_calibrate_and_model_pbl(
    infilename,
    incalfilename=None,
    refmodelname=None,
    residfilename=None,
    modelfilename=None,
    filteredfilename=None,
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
        model, resid, filtered, gains, fitted_info = calibrate_and_model_dpss(
            uvdata=uvdata, sky_model=sky_model, gains=gains, **cal_kwargs
        )
    else:
        raise NotImplementedError("only 'dpss' modeling basis is implemented.")
    if residfilename is not None:
        resid.write_uvh5(residfilename, clobber=clobber)
    if modelfilename is not None:
        model.write_uvh5(modelfilename, clobber=clobber)
    if filteredfilename is not None:
        filtered.write_uvh5(filteredfilename, clobber=clobber)
    if calfilename is not None:
        gains.write_calfits(calfilename, clobber=clobber)
    return model, resid, filtered, gains, fitted_info


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
    io_opts.add_argument(
        "--filteredfilename",
        type=str,
        help="Path to write output uvh5 filtered and calibrated data.",
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
