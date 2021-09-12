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
import re


OPTIMIZERS = {
    "Adadelta": tf.optimizers.Adadelta,
    "Adam": tf.optimizers.Adam,
    "Adamax": tf.optimizers.Adamax,
    "Ftrl": tf.optimizers.Ftrl,
    "Nadam": tf.optimizers.Nadam,
    "SGD": tf.optimizers.SGD,
    "RMSprop": tf.optimizers.RMSprop,
}


def chunk_fg_comp_dict_by_nbls(fg_model_comps_dict, use_redundancy=False, grp_size_threshold=5):
    """
    Order dict keys in order of number of baselines in each group


    chunk fit_groups in fg_model_comps_dict into chunks where all groups in the
    same chunk have the same number of baselines in each group.

    Parameters
    ----------
    fg_model_comps_dict: dict
        dictionary with keys that are tuples of tuples of 2-tuples (thats right, 3 levels)
        in the first level, each tuple represents a 'modeling group' visibilities in each
        modeling group are represented by a set of basis vectors that span all baselines in that
        group with elements raveled by baseline and then frequency. Each tuple in the modeling group is a
        'redundant group' representing visibilities that we will represent with identical component coefficients
        each element of each 'redundant group' is a 2-tuple antenna pair. Our formalism easily accomodates modeling
        visibilities as redundant or non redundant (one simply needs to make each redundant group length 1).

    use_redundancy: bool, optional
        If False, break fitting groups with the same number of baselines in each redundant
        sub_group into different fitting groups with no redundancy in each
        redundant subgroup. This is to prevent fitting groups with single
        redundant groups of varying lengths from being lumped into different chunks
        increasing the number of chunks has a more significant impact on run-time
        then increasing the number of baselines in each chunk.
        default is False.
    Returns:
    fg_model_comps_dict_chunked: dict
        dictionary where each key is a 2-tuple (nbl, nvecs) referring to the number
        of baselines in each vector and the number of vectors. Each 2-tuple points to
        a dictionary where each key is the fitting group in fg_comps_dict that includes
        nbl baselines. Each key in the referenced dict points to an (nred_grps * nfreqs x nvecs)
        numpy.ndarray describing the modeling components for each fitting group in the chunk.

    """
    chunked_keys = {}
    maxvecs = {}
    fg_model_comps_dict = copy.deepcopy(fg_model_comps_dict)
    if not use_redundancy:
        # We can remove redundancies for fitting groups of baselines that have the same
        # number of elements in each redundant group.
        keys_with_redundancy = list(fg_model_comps_dict.keys())
        for fit_grp in keys_with_redundancy:
            rlens = np.asarray([len(red_grp) for red_grp in fit_grp])
            # only break up groups with small numbers of group elements.
            if np.allclose(rlens, np.mean(rlens)) and len(rlens) < grp_size_threshold:
                # split up groups.
                modeling_vectors = fg_model_comps_dict.pop(fit_grp)
                for rednum in range(int(rlens[0])):
                    fit_grp_new = tuple([(red_grp[rednum],) for red_grp in fit_grp])
                    fg_model_comps_dict[fit_grp_new] = modeling_vectors

    for fit_grp in fg_model_comps_dict:
        nbl = 0
        for red_grp in fit_grp:
            for ap in red_grp:
                nbl += 1
        if nbl in chunked_keys:
            chunked_keys[nbl].append(fit_grp)
            if fg_model_comps_dict[fit_grp].shape[1] > maxvecs[nbl]:
                maxvecs[nbl] = fg_model_comps_dict[fit_grp].shape[1]
        else:
            chunked_keys[nbl] = [fit_grp]
            maxvecs[nbl] = fg_model_comps_dict[fit_grp].shape[1]

    fg_model_comps_dict_chunked = {}

    for nbl in chunked_keys:
        fg_model_comps_dict_chunked[(nbl, maxvecs[nbl])] = {k: fg_model_comps_dict[k] for k in chunked_keys[nbl]}

    return fg_model_comps_dict_chunked


def tensorize_fg_model_comps_dict(
    fg_model_comps_dict,
    ants_map,
    nfreqs,
    use_redundancy=False,
    dtype=np.float32,
    notebook_progressbar=False,
    verbose=False,
    grp_size_threshold=5,
):
    """Convert per-baseline model components into a Ndata x Ncomponent tensor

    Parameters
    ----------
    fg_model_comps_dict: dict
        dictionary where each key is a 2-tuple (nbl, nvecs) referring to the number
        of baselines in each vector and the number of vectors. Each 2-tuple points to
        a dictionary where each key is the fitting group in fg_comps_dict that includes
        nbl baselines. Each key in the referenced dict points to an (nred_grps * nfreqs x nvecs)
        numpy.ndarray describing the modeling components for each fitting group in the chunk.
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    nfreqs: int, optional
        number of frequency channels
    dtype: numpy.dtype
        tensor data types
        default is np.float32

    Returns
    -------
    fg_model_comps: list
        list of tf.Tensor objects where each tensor has shape (nvecs, ngrps, nbls, nfreqs)
        where nbls varies from tensor to tensor. Fitting groups with vectors that span nbls are lumped into the same
        modeling tensor along the ngrps axis. nvecs is chosen in chunk_fg_comp_dict_by_nbls
        to be the maximum number of vectors representing any of the ngrps baseline grps
        which means that many rows in nvecs will be zero. For example, if we are modeling with
        vectors that all span nbls=1 baseline and using delay-modes to model our data
        then nvecs will equal the largest number of delay modes necessary to model the wedge
        on all baselines even though the short baselines are described by far fewer modes
        on short baselines, most of the rows along the vector dimension will therefor be zero.
        This is wasteful of memory but it allows us to take advantage of the fast
        dense matrix operations on a GPU.

    corr_inds: list
        list of list of lists of 2-tuples. Hierarchy of lists is
        chunk
            group
                baseline - (int 2-tuple)

    """
    echo(
        f"{datetime.datetime.now()} Computing foreground components matrices...\n",
        verbose=verbose,
    )
    # chunk foreground components.
    fg_model_comps_dict = chunk_fg_comp_dict_by_nbls(
        fg_model_comps_dict, use_redundancy=use_redundancy, grp_size_threshold=grp_size_threshold
    )
    fg_model_comps = []
    corr_inds = []
    for nbls, nvecs in fg_model_comps_dict:
        ngrps = len(fg_model_comps_dict[(nbls, nvecs)])
        modeling_matrix = np.zeros((nvecs, ngrps, nbls, nfreqs))

        corr_inds_chunk = []
        for grpnum, modeling_grp in enumerate(fg_model_comps_dict[(nbls, nvecs)]):
            corr_inds_grp = []
            nbl = 0
            for rgrpnum, red_grp in enumerate(modeling_grp):
                nred = len(red_grp)
                for ap in red_grp:
                    i, j = ants_map[ap[0]], ants_map[ap[1]]
                    corr_inds_grp.append((i, j))
                    vecslice = slice(0, fg_model_comps_dict[(nbls, nvecs)][modeling_grp].shape[1])
                    compslice = slice(rgrpnum * nfreqs, (rgrpnum + 1) * nfreqs)
                    dslice = slice(nbl * nfreqs, (nbl + 1) * nfreqs)
                    modeling_matrix[vecslice, grpnum, nbl] = fg_model_comps_dict[(nbls, nvecs)][modeling_grp][
                        compslice
                    ].T
                    nbl += 1
            corr_inds_chunk.append(corr_inds_grp)

        fg_model_comps.append(tf.convert_to_tensor(modeling_matrix, dtype=dtype))
        corr_inds.append(corr_inds_chunk)

    return fg_model_comps, corr_inds


def tensorize_data(
    uvdata,
    corr_inds,
    ants_map,
    polarization,
    time_index,
    data_scale_factor=1.0,
    weights=None,
    nsamples_in_weights=False,
    dtype=np.float32,
):
    """Convert data in uvdata object to a tensor

    Parameters
    ----------
    uvdata: UVData object
        UVData object containing data, flags, and nsamples to tensorize.
    corr_inds: list
        list of list of lists of 2-tuples. Hierarchy of lists is
        chunk
            group
                baseline - (int 2-tuple)
    ants_map: dict mapping integers to integers
        map between each antenna number to a unique index between 0 and Nants_data
        (typically the index of each antenna in ants_map)
    polarization: str
        pol-str of gain to extract.
    time_index: int
        index of time to convert to tensors
    data_scale_factor: float, optional
        overall scaling factor to divide tensorized data by.
        default is 1.0
    weights: UVFlag object, optional
        UVFlag weights object containing weights to use for data fitting.
        default is None -> use nsamples * ~flags if nsamples_in_weights
        or ~flags if not nsamples_in_weights
    nsamples_in_weights: bool, optional
        If True and weights is None, generate weights proportional to nsamples.
        default is False.
    dtype: numpy.dtype
        data-type to store in tensor.
        default is np.float32

    Returns
    -------
    data_r: list of tf.Tensor objects
        list of tf.Tensor objects. Each tensor has shape (ngrps, nbls, nfreqs)
        where ngrps, nbls are the dimensions of each sublist in corr_inds
        and contain the real components of the baselines specified by these 2-tuples.
    data_i: list of tf.Tensor objects
        list of tf.Tensor objects. Each tensor has shape (ngrps, nbls, nfreqs)
        where ngrps, nbls are the dimensions of each sublist in corr_inds
        and contain the imag components of the baselines specified by these 2-tuples.
    wgts: tf.Tensor object
        list of tf.Tensor objects. Each tensor has shape (ngrps, nbls, nfreqs)
        where ngrps, nbls are the dimensions of each sublist in corr_inds
        and contain the weights of the baselines specified by these 2-tuples.
    """
    ants_map_inv = {ants_map[i]: i for i in ants_map}
    dshape = (uvdata.Nants_data, uvdata.Nants_data, uvdata.Nfreqs)
    data_r = np.zeros(dshape, dtype=dtype)
    data_i = np.zeros_like(data_r)
    wgts = np.zeros_like(data_r)
    wgtsum = 0.0
    for chunk in corr_inds:
        for fitgrp in chunk:
            for (i, j) in fitgrp:
                ap = ants_map_inv[i], ants_map_inv[j]
                bl = ap + (polarization,)
                data = uvdata.get_data(bl)[time_index] / data_scale_factor
                iflags = (~uvdata.get_flags(bl))[time_index].astype(dtype)
                nsamples = uvdata.get_nsamples(bl)[time_index].astype(dtype)
                data_r[i, j] = data.real.astype(dtype)
                data_i[i, j] = data.imag.astype(dtype)
                if weights is None:
                    if nsamples_in_weights:
                        wgts[i, j] = iflags * nsamples
                    else:
                        wgts[i, j] = iflags
                else:
                    if ap in weights.get_antpairs():
                        dinds = weights.antpair2ind(*ap)
                    else:
                        dinds = weights.antpair2ind(*ap[::-1])
                    polnum = np.where(
                        weights.polarization_array
                        == uvutils.polstr2num(polarization, x_orientation=weights.x_orientation)
                    )[0][0]
                    wgts[i, j] = weights.weights_array[dinds[time_index], 0, :, polnum].astype(dtype) * iflags
                wgtsum += np.sum(wgts[i, j])
    data_r = tf.convert_to_tensor(data_r, dtype=dtype)
    data_i = tf.convert_to_tensor(data_i, dtype=dtype)
    wgts = tf.convert_to_tensor(wgts / wgtsum, dtype=dtype)

    nchunks = len(corr_inds)
    data_r = [tf.gather_nd(data_r, corr_inds[cnum]) for cnum in range(nchunks)]
    data_i = [tf.gather_nd(data_i, corr_inds[cnum]) for cnum in range(nchunks)]
    wgts = [tf.gather_nd(wgts, corr_inds[cnum]) for cnum in range(nchunks)]

    return data_r, data_i, wgts



def renormalize(uvdata_reference_model, uvdata_deconv, gains, polarization, time_index, additional_flags=None):

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
    additional_flags: np.ndarray
        Any additional flags you wish to use for excluding data from normalization
        fed as an np.ndarray with same shape as uvdata_reference_model and uvdata_deconv.
        default is None -> Only exclude data in flags from reference model and deconv from
        determinging normalization.
    Returns
    -------
    N/A: Modifies uvdata_deconv and gains in-place.
    """
    # compute and multiply out scale-factor accounting for overall amplitude and phase degeneracy.
    polnum_data = np.where(
        uvdata_deconv.polarization_array == uvutils.polstr2num(polarization, x_orientation=uvdata_deconv.x_orientation)
    )[0][0]


    bltsel = uvdata_deconv.time_array == np.unique(uvdata_deconv.time_array)[time_index]

    selection = (
        ~uvdata_deconv.flag_array[bltsel, :, :, polnum_data]
        & ~uvdata_reference_model.flag_array[bltsel, :, :, polnum_data]
    )
    if additional_flags is not None:
        selection = selection & ~additional_flags[bltsel, :, :, polnum_data]

    data_ratio = (
        uvdata_reference_model.data_array[bltsel, :, :, polnum_data][selection]
        / uvdata_deconv.data_array[bltsel, :, :, polnum_data][selection]
    )

    data_ratio[~np.isfinite(data_ratio)] = np.nan

    scale_factor_phase = np.angle(np.nanmean(data_ratio))
    scale_factor_abs = np.sqrt(np.nanmean(np.abs(data_ratio) ** 2.0))
    scale_factor = scale_factor_abs * np.exp(1j * scale_factor_phase)
    uvdata_deconv.data_array[bltsel, :, :, polnum_data] *= scale_factor

    polnum_gains = np.where(
        gains.jones_array == uvutils.polstr2num(polarization, x_orientation=uvdata_deconv.x_orientation)
    )[0][0]
    gains.gain_array[:, :, :, time_index, polnum_data] *= (scale_factor) ** -0.5


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
    fg_model_comps,
    fg_coeffs,
    corr_inds,
):
    """Compute tensor foreground model.

    Parameters
    ----------
    nants: int
        number of antennas in data to model.
    freqs: int
        number of frequencies in data to model.
    fg_model_comps: list
        list of fg modeling tf.Tensor objects
        representing foreground modeling vectors.
        Each tensor is (nvecs, ngrps, nbls, nfreqs)
    fg_coeffs: list
        list of fg modeling tf.Tensor objects
        representing foreground modeling coefficients.
        Each tensor is (nvecs, ngrps, 1, 1)
    corr_inds: list
        list of list of lists of 2-tuples. Hierarchy of lists is
        chunk
            group
                baseline - (int 2-tuple)

    Returns
    -------
    model: tf.Tensor object
        nants x nants x nfreqs model of the visibility data
    """
    model = np.zeros((nants, nants, nfreqs))
    nchunks = len(fg_model_comps)
    for cnum in range(nchunks):
        ngrps = fg_model_comps[cnum].shape[1]
        gchunk = tf.reduce_sum(fg_coeffs[cnum] * fg_model_comps[cnum], axis=0).numpy()
        for gnum in range(ngrps):
            for blnum, (i, j) in enumerate(corr_inds[cnum][gnum]):
                model[i, j] = gchunk[gnum, blnum]
    return model


def fit_gains_and_foregrounds(
    g_r,
    g_i,
    fg_r,
    fg_i,
    data_r,
    data_i,
    wgts,
    fg_comps,
    corr_inds,
    use_min=False,
    tol=1e-14,
    maxsteps=10000,
    optimizer="Adamax",
    freeze_model=False,
    verbose=False,
    notebook_progressbar=False,
    dtype=np.float32,
    graph_mode=False,
    n_profile_steps=0,
    profile_log_dir="./logdir",
    sky_model_r=None,
    sky_model_i=None,
    model_regularization=None,
    graph_args_dict=None,
    **opt_kwargs,
):
    """Run optimization loop to fit gains and foreground components.

    Parameters
    ----------
    g_r: tf.Tensor object.
        tf.Tensor object holding real parts of gains.
    g_i: tf.Tensor object.
        tf.Tensor object holding imag parts of gains.
    fg_r: list
        list of tf.Tensor objects. Each has shape (nvecs, ngrps, 1, 1)
        tf.Tensor object holding foreground coeffs.
    fg_i: list
        list of tf.Tensor objects. Each has shape (nvecs, ngrps, 1, 1)
        tf.Tensor object holding imag coeffs.
    data_r: list
        list of tf.Tensor objects. Each has shape (ngrps, nbls, nfreqs)
        real part of data to fit.
    data_i: list
        list of tf.Tensor objects. Each has shape (ngrps, nbls, nfreqs)
        imag part of data to fit.
    wgts: list
        list of tf.Tensor objects. Each has shape (ngrps, nbls, nfreqs)
    fg_comps: list:
        list of tf.Tensor objects. Each has shape (nvecs, ngrps, nbls, nfreqs)
        represents vectors to be used in modeling visibilities.
    corr_inds: list
        list of list of lists of 2-tuples. Hierarchy of lists is
        chunk
            group
                baseline - (int 2-tuple)
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
    graph_mode: bool, optional
        if True, compile gradient update step in graph mode to speed up
        runtime by ~2-3x. I've found that this helps on CPUs but on GPUs
        it actually increases runtime by a similar factor.
    n_profile_steps: bool, optional
        number of steps to run profiling on
        default is 0.
    profile_log_dir: str, optional
        directory to save profile logs to
        default is './logdir'
    sky_model_r: list of tf.Tensor objects, optional
        chunked tensors containing model in same format as data_r
    sky_model_i: list of tf.Tensor objects, optional
        chunked tensors containing model in the same format as data_i
    model_regularization: str, optional
        type of model regularization to perform. Currently support "sum"
        where the sums of real and imaginary parts (across all bls and freqs)
        are constrained to be the same as the sum of real and imag parts
        of data.
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
    if graph_args_dict is None:
        graph_args_dict = {}
    # initialize the optimizer.
    echo(f"{datetime.datetime.now()} Provided the following opt_kwargs")
    for k in opt_kwargs:
        echo(f"{k}: {opt_kwargs[k]}")
    opt = OPTIMIZERS[optimizer](**opt_kwargs)
    # set up history recording
    fit_history = {"loss": []}
    min_loss = 9e99
    nants = g_r.shape[0]
    nfreqs = g_r.shape[1]
    ant0_inds = []
    ant1_inds = []
    nchunks = len(fg_comps)
    # build up list of lists of ant0 and ant1 for gather ops
    for cnum in range(nchunks):
        ant0_chunk = []
        ant1_chunk = []
        ngrps = len(corr_inds[cnum])
        for gnum in range(ngrps):
            ant0_grp = []
            ant1_grp = []
            for cpair in corr_inds[cnum][gnum]:
                ant0_grp.append(cpair[0])
                ant1_grp.append(cpair[1])
            ant0_chunk.append(ant0_grp)
            ant1_chunk.append(ant1_grp)
        ant0_inds.append(ant0_chunk)
        ant1_inds.append(ant1_chunk)

    g_r = tf.Variable(g_r)
    g_i = tf.Variable(g_i)
    if not freeze_model:
        fg_r = [tf.Variable(fgr) for fgr in fg_r]
        fg_i = [tf.Variable(fgi) for fgi in fg_i]
        vars = [g_r, g_i] + fg_r + fg_i
    else:
        vars = [g_r, g_i]

    echo(
        f"{datetime.datetime.now()} Performing gradient descent on {np.prod(g_r.shape)} complex gain parameters...",
        verbose=verbose,
    )
    if not freeze_model:
        echo(
            f"Performing gradient descent on total of {int(np.sum([fgr.shape[0] * fgr.shape[1] for fgr in fg_r]))} complex foreground parameters",
            verbose=verbose,
        )
        echo(
            f"Foreground Parameters grouped into chunks of shape ((nvecs, ngrps): nbls) {[str(fgr.shape[:2]) + ':' + str(dc.shape[1]) for fgr, dc in zip(fg_r, data_r)]}",
            verbose=verbose,
        )

    if model_regularization == "sum":
        prior_r_sum = tf.reduce_sum(
            tf.stack([tf.reduce_sum(sky_model_r[cnum] * wgts[cnum]) for cnum in range(nchunks)])
        )
        prior_i_sum = tf.reduce_sum(
            tf.stack([tf.reduce_sum(sky_model_i[cnum] * wgts[cnum]) for cnum in range(nchunks)])
        )

        def loss_function():
            return mse_chunked_sum_regularized(
                g_r=g_r,
                g_i=g_i,
                fg_r=fg_r,
                fg_i=fg_i,
                fg_comps=fg_comps,
                nchunks=nchunks,
                data_r=data_r,
                data_i=data_i,
                wgts=wgts,
                ant0_inds=ant0_inds,
                ant1_inds=ant1_inds,
                dtype=dtype,
                prior_r_sum=prior_r_sum,
                prior_i_sum=prior_i_sum,
            )

    else:

        def loss_function():
            return mse_chunked(
                g_r=g_r,
                g_i=g_i,
                fg_r=fg_r,
                fg_i=fg_i,
                fg_comps=fg_comps,
                nchunks=nchunks,
                data_r=data_r,
                data_i=data_i,
                wgts=wgts,
                ant0_inds=ant0_inds,
                ant1_inds=ant1_inds,
                dtype=dtype,
            )

    def train_step_code():
        with tf.GradientTape() as tape:
            loss = loss_function()
        grads = tape.gradient(loss, vars)
        opt.apply_gradients(zip(grads, vars))
        return loss

    if graph_mode:

        @tf.function(**graph_args_dict)
        def train_step():
            return train_step_code()

    else:

        def train_step():
            return train_step_code()

    if n_profile_steps > 0:
        echo(f"{datetime.datetime.now()} Profiling with {n_profile_steps}. And writing output to {profile_log_dir}...")
        tf.profiler.experimental.start(profile_log_dir)
        for step in PBARS[notebook_progressbar](range(n_profile_steps)):
            with tf.profiler.experimental.Trace("train", step_num=step):
                train_step()
        tf.profiler.experimental.stop()

    echo(
        f"{datetime.datetime.now()} Building Computational Graph...\n",
        verbose=verbose,
    )
    loss = train_step()
    echo(
        f"{datetime.datetime.now()} Performing Gradient Descent. Initial MSE of {loss:.2e}...\n",
        verbose=verbose,
    )

    for step in PBARS[notebook_progressbar](range(maxsteps)):
        loss = train_step()
        fit_history["loss"].append(loss.numpy())
        if use_min and fit_history["loss"][-1] < min_loss:
            # store the g_r, g_i, fg_r, fg_i values that minimize loss
            # in case of overshoot.
            min_loss = fit_history["loss"][-1]
            g_r_opt = g_r.value()
            g_i_opt = g_i.value()
            if not freeze_model:
                fg_r_opt = [fgr.value() for fgr in fg_r]
                fg_i_opt = [fgi.value() for fgi in fg_i]

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
            fg_r_opt = [fgr.value() for fgr in fg_r]
            fg_i_opt = [fgi.value() for fgi in fg_i]

        else:
            fg_r_opt = fg_r
            fg_i_opt = fg_i

    echo(
        f"{datetime.datetime.now()} Finished Gradient Descent. MSE of {min_loss:.2e}...\n",
        verbose=verbose,
    )
    return g_r_opt, g_i_opt, fg_r_opt, fg_i_opt, fit_history


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
    """Insert fitted tensor values back into uvdata object for tensor mode.

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
    data,
    wgts,
    fg_model_comps,
    notebook_progressbar=False,
    verbose=False,
):
    """Initialize foreground coefficient tensors from uvdata and modeling component dictionaries.


    Parameters
    ----------
    data: list
        list of tf.Tensor objects, each with shape (ngrps, nbls, nfreqs)
        representing data
    wgts: list
        list of tf.Tensor objects, each with shape (ngrps, nbls, nfreqs)
        representing weights.
    fg_model_comps: list
        list of fg modeling tf.Tensor objects
        representing foreground modeling vectors.
        Each tensor is (nvecs, ngrps, nbls, nfreqs)
        see description in tensorize_fg_model_comps_dict
        docstring.
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    verbose: bool, optional
        lots of text output
        default is False.
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
    echo(
        f"{datetime.datetime.now()} Computing initial foreground coefficient guesses using linear-leastsq...\n",
        verbose=verbose,
    )
    fg_coeffs = []
    nchunks = len(data)
    binary_wgts = [
        tf.convert_to_tensor(~np.isclose(wgts[cnum].numpy(), 0.0), dtype=wgts[cnum].dtype) for cnum in range(nchunks)
    ]
    for cnum in PBARS[notebook_progressbar](range(nchunks)):
        # set up linear leastsq
        fg_coeff_chunk = []
        ngrps = data[cnum].shape[0]
        ndata = data[cnum].shape[1] * data[cnum].shape[2]
        nvecs = fg_model_comps[cnum].shape[0]
        # pad with zeros
        for gnum in range(ngrps):
            nonzero_rows = np.where(
                np.all(np.isclose(fg_model_comps[cnum][:, gnum].numpy().reshape(nvecs, ndata), 0.0), axis=1)
            )[0]
            if len(nonzero_rows) > 0:
                nvecs_nonzero = np.min(nonzero_rows)
            else:
                nvecs_nonzero = nvecs
            # solve linear leastsq
            fg_coeff_chunk.append(
                tf.reshape(
                    tf.linalg.lstsq(
                        tf.transpose(tf.reshape(fg_model_comps[cnum][:, gnum], (nvecs, ndata)))[:, :nvecs_nonzero],
                        tf.reshape(data[cnum][gnum] * binary_wgts[cnum][gnum], (ndata, 1)),
                    ),
                    (nvecs_nonzero,),
                )
            )
            # pad zeros at the end back up to nvecs.
            fg_coeff_chunk[-1] = tf.pad(fg_coeff_chunk[-1], [(0, nvecs - nvecs_nonzero)])
        # add two additional dummy indices to satify broadcasting rules.
        fg_coeff_chunk = tf.reshape(tf.transpose(tf.stack(fg_coeff_chunk)), (nvecs, ngrps, 1, 1))
        fg_coeffs.append(fg_coeff_chunk)

    echo(
        f"{datetime.datetime.now()} Finished initial foreground coefficient guesses...\n",
        verbose=verbose,
    )
    return fg_coeffs


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
    nsamples_in_weights=True,
    graph_mode=False,
    grp_size_threshold=5,
    n_profile_steps=0,
    profile_log_dir="./logdir",
    model_regularization="post_hoc",
    init_guesses_from_previous_time_step=True,
    skip_threshold=0.5,
    **opt_kwargs,
):
    """Perform simultaneous calibration and foreground fitting using tensors.


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
        default is None -> use nsamples * ~flags if nsamples_in_weights
        or ~flags if not nsamples_in_weights
    nsamples_in_weights: bool, optional
        If True and weights is None, generate weights proportional to nsamples.
        default is True.
    graph_mode: bool, optional
        if True, compile gradient update step in graph mode to speed up
        runtime by ~2-3x. I've found that this helps on CPUs but on GPUs
        it actually increases runtime by a similar factor.
    n_profile_steps: bool, optional
        number of steps to run profiling on
        default is 0.
    profile_log_dir: str, optional
        directory to save profile logs to
        default is './logdir'
    model_regularization: str, optional
        option to regularize model
        supported 'post_hoc', 'sum'
        default is 'post_hoc'
        which sets sum of amps equal and sum of phases equal.
    init_guesses_from_previous_time_step: bool, optional
        if True, then use foreground coeffs and gains from previous time-step to
        initialize gains for next time step.
    skip_threshold: float, optional
        if less then this fraction of data is unflagged on a particular poltime,
        flag the entire poltime.
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
    model.data_array[:] = 0.0
    model.flag_array[:] = False

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

    if sky_model is None and model_regularization is not None:
        echo(
            f"{datetime.datetime.now()} Sky model is None. Initializing from data...\n",
            verbose=verbose,
        )
        sky_model = cal_utils.apply_gains(uvdata, gains)
    else:
        sky_model = sky_model.select(inplace=False, bls=[ap for ap in antpairs_data])

    fit_history = {}
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    # generate tensors to hold foreground components.
    fg_model_comps, corr_inds = tensorize_fg_model_comps_dict(
        fg_model_comps_dict=fg_model_comps_dict,
        ants_map=ants_map,
        dtype=dtype,
        nfreqs=sky_model.Nfreqs,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        use_redundancy=use_redundancy,
        grp_size_threshold=grp_size_threshold,
    )
    echo(
        f"{datetime.datetime.now()}Finished Converting Foreground Modeling Components to Tensors...\n",
        verbose=verbose,
    )
    # delete fg_model_comps_dict. It can take up a lot of memory.
    del fg_model_comps_dict
    # loop through polarization and times.
    for polnum, pol in enumerate(uvdata.get_pols()):
        echo(
            f"{datetime.datetime.now()} Working on pol {pol}, {polnum + 1} of {uvdata.Npols}...\n",
            verbose=verbose,
        )
        fit_history_p = {}
        first_time = True
        for time_index in range(uvdata.Ntimes):
            echo(
                f"{datetime.datetime.now()} Working on time {time_index + 1} of {uvdata.Ntimes}...\n",
                verbose=verbose,
            )
            bltsel = uvdata.time_array == np.unique(uvdata.time_array)[time_index]
            frac_unflagged = np.count_nonzero(~uvdata.flag_array[bltsel, 0, :, polnum]) / (
                uvdata.Ntimes * uvdata.Nfreqs
            )
            # check that fraction of unflagged data > skip_threshold.
            if frac_unflagged >= skip_threshold:
                rmsdata = np.sqrt(
                    np.mean(
                        np.abs(uvdata.data_array[bltsel, 0, :, polnum][~uvdata.flag_array[bltsel, 0, :, polnum]])
                        ** 2.0
                    )
                )
                echo(f"{datetime.datetime.now()} Tensorizing data...\n", verbose=verbose)
                data_r, data_i, wgts = tensorize_data(
                    uvdata,
                    corr_inds=corr_inds,
                    ants_map=ants_map,
                    polarization=pol,
                    time_index=time_index,
                    data_scale_factor=rmsdata,
                    weights=weights,
                    nsamples_in_weights=nsamples_in_weights,
                    dtype=dtype,
                )

                if sky_model is not None:
                    echo(f"{datetime.datetime.now()} Tensorizing sky model...\n", verbose=verbose)
                    sky_model_r, sky_model_i, _ = tensorize_data(
                        sky_model,
                        corr_inds=corr_inds,
                        ants_map=ants_map,
                        polarization=pol,
                        time_index=time_index,
                        data_scale_factor=rmsdata,
                        weights=weights,
                        dtype=dtype,
                    )
                else:
                    sky_model_r, sky_model_i = None, None
                if first_time or not init_guesses_from_previous_time_step:
                    first_time = False
                    echo(f"{datetime.datetime.now()} Tensorizing Gains...\n", verbose=verbose)
                    g_r, g_i = tensorize_gains(gains, dtype=dtype, time_index=time_index, polarization=pol)
                    # generate initial guess for foreground coeffs.
                    echo(
                        f"{datetime.datetime.now()} Tensorizing Foreground coeffs...\n",
                        verbose=verbose,
                    )
                    fg_r = tensorize_fg_coeffs(
                        data=data_r,
                        wgts=wgts,
                        fg_model_comps=fg_model_comps,
                        verbose=verbose,
                        notebook_progressbar=notebook_progressbar,
                    )

                    fg_i = tensorize_fg_coeffs(
                        data=data_i,
                        wgts=wgts,
                        fg_model_comps=fg_model_comps,
                        verbose=verbose,
                        notebook_progressbar=notebook_progressbar,
                    )

                (g_r, g_i, fg_r, fg_i, fit_history_p[time_index],) = fit_gains_and_foregrounds(
                    g_r=g_r,
                    g_i=g_i,
                    fg_r=fg_r,
                    fg_i=fg_i,
                    data_r=data_r,
                    data_i=data_i,
                    wgts=wgts,
                    fg_comps=fg_model_comps,
                    corr_inds=corr_inds,
                    optimizer=optimizer,
                    use_min=use_min,
                    freeze_model=freeze_model,
                    notebook_progressbar=notebook_progressbar,
                    verbose=verbose,
                    tol=tol,
                    dtype=dtype,
                    maxsteps=maxsteps,
                    graph_mode=graph_mode,
                    n_profile_steps=n_profile_steps,
                    profile_log_dir=profile_log_dir,
                    sky_model_r=sky_model_r,
                    sky_model_i=sky_model_i,
                    model_regularization=model_regularization,
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
                        fg_model_comps=fg_model_comps,
                        fg_coeffs=fg_r,
                        corr_inds=corr_inds,
                        nants=uvdata.Nants_data,
                        nfreqs=uvdata.Nfreqs,
                    ),
                    model_i=yield_fg_model_array(
                        fg_model_comps=fg_model_comps,
                        fg_coeffs=fg_i,
                        corr_inds=corr_inds,
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
                    gains_re=g_r,
                    gains_im=g_i,
                )
            else:
                echo(
                    f"{datetime.datetime.now()}: Only {frac_unflagged * 100}-percent of data unflagged. Skipping...\n",
                    verbose=verbose,
                )
                flag_poltime(resid, time_index=time_index, polarization=pol)
                flag_poltime(gains, time_index=time_index, polarization=pol)
                flag_poltime(model, time_index=time_index, polarization=pol)
                fit_history[polnum] = "skipped!"
            # normalize on sky model if we use post-hoc regularization
            if not freeze_model and model_regularization == "post_hoc" and np.any(~model.flag_array[bltsel]):
                renormalize(
                    uvdata_reference_model=sky_model,
                    uvdata_deconv=model,
                    gains=gains,
                    polarization=pol,
                    time_index=time_index,
                    additional_flags=uvdata.flag_array,
                )
        fit_history[polnum] = fit_history_p

    model_with_gains = cal_utils.apply_gains(model, gains, inverse=True)
    if not correct_model:
        model = model_with_gains
    resid.data_array -= model_with_gains.data_array
    resid.data_array[model_with_gains.flag_array] = 0.0  # set resid to zero where model is flagged.
    resid.data_array[uvdata.flag_array] = 0.0  # also set resid to zero where data is flagged.
    if correct_resid:
        resid = cal_utils.apply_gains(resid, gains)

    return model, resid, gains, fit_history


def flag_poltime(data_object, time_index, polarization):
    if isinstance(data_object, UVData):
        bltsel = data_object.time_array == np.unique(data_object.time_array)[time_index]
        polnum = np.where(
            data_object.polarization_array == uvutils.polstr2num(polarization, x_orientation=data_object.x_orientation)
        )[0][0]
        data_object.flag_array[bltsel, :, :, polnum] = True
        data_object.data_array[bltsel, :, :, polnum] = 0.0
    elif isinstance(data_object, UVCal):
        polnum = np.where(
            data_object.jones_array == uvutils.polstr2num(polarization, x_orientation=data_object.x_orientation)
        )[0][0]
        data_object.gain_array[:, 0, :, time_index, polnum] = 1.0
        data_object.flag_array[:, 0, :, time_index, polnum] = True
    else:
        raise ValueError("only supports data_object that is UVCal or UVData.")


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
    grp_size_threshold=5,
    model_comps_dict=None,
    save_dict_to=None,
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
    grp_size_threshold: int, optional
      groups with number of elements less then this value are split up into single baselines.
      default is 5.
    model_comps_dict: dict, optional
        dictionary mapping fitting groups to numpy.ndarray see modeling.yield_mixed_comps
        for more specifics.
        default is None -> compute fitting groups automatically.
    save_dict_to: str, optional
        save model_comps_dict to hdf5 container if True
        default is False.
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
        red_tol=red_tol,
        include_autos=include_autos,
        red_tol_freq=red_tol_freq,
        n_angle_bins=n_angle_bins,
        notebook_progressbar=notebook_progressbar,
        require_exact_angle_match=require_exact_angle_match,
        angle_match_tol=angle_match_tol,
    )

    if model_comps_dict is None:
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
            grp_size_threshold=grp_size_threshold,
        )

    if save_dict_to is not None:
        np.save(save_dict_to, model_comps_dict)

    (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
        uvdata=uvdata,
        fg_model_comps_dict=model_comps_dict,
        include_autos=include_autos,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        use_redundancy=use_redundancy,
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
    red_tol=1.0,
    notebook_progressbar=False,
    fg_model_comps_dict=None,
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
    red_tol: float, optional
        tolerance for treating baselines as redundant (meters)
        default is 1.0
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    fg_model_comps_dict: dict, optional
        dictionary containing precomputed foreground model components.
        Currently only supported if use_redundancy is False.
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
        verbose=verbose,
    )

    (model, resid, gains, fitted_info,) = calibrate_and_model_tensor(
        uvdata=uvdata,
        fg_model_comps_dict=dpss_model_comps_dict,
        include_autos=include_autos,
        verbose=verbose,
        notebook_progressbar=notebook_progressbar,
        **fitting_kwargs,
    )

    return model, resid, gains, fitted_info


def fg_model(fg_r, fg_i, fg_comps):
    vr = tf.reduce_sum(fg_r * fg_comps, axis=0)
    vi = tf.reduce_sum(fg_i * fg_comps, axis=0)
    return vr, vi


def data_model(g_r, g_i, fg_r, fg_i, fg_comps, ant0_inds, ant1_inds):
    gr0 = tf.gather(g_r, ant0_inds)
    gr1 = tf.gather(g_r, ant1_inds)
    gi0 = tf.gather(g_i, ant0_inds)
    gi1 = tf.gather(g_i, ant1_inds)
    grgr = gr0 * gr1
    gigi = gi0 * gi1
    grgi = gr0 * gi1
    gigr = gi0 * gr1
    vr, vi = fg_model(fg_r, fg_i, fg_comps)
    model_r = (grgr + gigi) * vr + (grgi - gigr) * vi
    model_i = (gigr - grgi) * vr + (grgr + gigi) * vi
    return model_r, model_i


def mse(model_r, model_i, data_r, data_i, wgts):
    return tf.reduce_sum((tf.square(data_r - model_r) + tf.square(data_i - model_i)) * wgts)


def mse_chunked(g_r, g_i, fg_r, fg_i, fg_comps, nchunks, data_r, data_i, wgts, ant0_inds, ant1_inds, dtype=np.float32):
    cal_loss = [tf.constant(0.0, dtype) for cnum in range(nchunks)]
    # now deal with dense components
    for cnum in range(nchunks):
        model_r, model_i = data_model(
            g_r, g_i, fg_r[cnum], fg_i[cnum], fg_comps[cnum], ant0_inds[cnum], ant1_inds[cnum]
        )
        cal_loss[cnum] += mse(model_r, model_i, data_r[cnum], data_i[cnum], wgts[cnum])
    return tf.reduce_sum(tf.stack(cal_loss))


def mse_chunked_sum_regularized(
    g_r,
    g_i,
    fg_r,
    fg_i,
    fg_comps,
    nchunks,
    data_r,
    data_i,
    wgts,
    ant0_inds,
    ant1_inds,
    prior_r_sum,
    prior_i_sum,
    dtype=np.float32,
):
    cal_loss = [tf.constant(0.0, dtype) for cnum in range(nchunks)]
    model_i_sum = [tf.constant(0.0, dtype) for cnum in range(nchunks)]
    model_r_sum = [tf.constant(0.0, dtype) for cnum in range(nchunks)]
    # now deal with dense components
    for cnum in range(nchunks):
        model_r, model_i = data_model(
            g_r, g_i, fg_r[cnum], fg_i[cnum], fg_comps[cnum], ant0_inds[cnum], ant1_inds[cnum]
        )
        # compute sum of real and imag parts x weights for regularization.
        model_r_sum[cnum] += tf.reduce_sum(model_r * wgts[cnum])
        model_i_sum[cnum] += tf.reduce_sum(model_i * wgts[cnum])

        cal_loss[cnum] += mse(model_r, model_i, data_r[cnum], data_i[cnum], wgts[cnum])
    return (
        tf.reduce_sum(tf.stack(cal_loss))
        + tf.square(tf.reduce_sum(tf.stack(model_r_sum)) - prior_r_sum)
        + tf.square(tf.reduce_sum(tf.stack(model_i_sum)) - prior_i_sum)
    )


def read_calibrate_and_model_dpss(
    input_data_files,
    input_model_files=None,
    input_gain_files=None,
    resid_outfilename=None,
    gain_outfilename=None,
    model_outfilename=None,
    output_directory="./",
    fitted_info_outfilename=None,
    x_orientation="east",
    clobber=False,
    bllen_min=0.0,
    bllen_max=np.inf,
    bl_ew_min=0.0,
    ex_ants=None,
    gpu_index=None,
    gpu_memory_limit=None,
    **calibration_kwargs,
):
    """
    Driver function for using calamity with DPSS modeling.

    Parameters
    ----------
    input_data_files: list of strings or UVData object.
        list of paths to input files to read in and calibrate.
    input_model_files: list of strings or UVData object, optional
        list of paths to model files for overal phase/amp reference.
        Default is None -> use input files as model for overall
        phase and amplitude calibration.
    input_gain_files: list of strings or UVCal object, optional
        list of paths to gain files to use as initial guesses for calibration.
    resid_outfilename: str, optional
        path for file to write residuals.
        default is None -> don't write out residuals.
    gain_outfilename: str, optional
        path to gain calfits to write fitted gains.
        default is None -> don't write out gains.
    model_outfilename, str, optional
        path to file to write model output.
        default is None -> Don't write model.
    fitting_info_outfilename, str, optional
        string to pickel fitting info to.
    n_output_chunks: int optional
        split up outputs into n_output_chunks chunked by time.
        default is None -> write single output file.
    bllen_min: float, optional
        select all baselines with length greater then this value [meters].
        default is 0.0
    bllen_max: float, optional
        select only baselines with length less then this value [meters].
        default is np.inf.
    bl_ew_min: float, optional
        select all baselines with EW projected length greater then this value [meters].
        default is 0.0
    gpu_index: int, optional
        limit visible GPUs to be the index of this GPU.
        default: None -> all GPUs are visible.
    gpu_memory_limit: float, optional
        GiB of memory on GPU that can be used.
        default None -> all memory available.
    calibration_kwargs: kwarg dict
        see kwrags for calibration_and_model_dpss()
    Returns
    -------

    model_fit: UVData object
        uvdata object containing DPSS model of intrinsic foregrounds.
    resid_fit: UVData object
        uvdata object containing residuals after subtracting model times gains and applying gains.
    gains_fit: UVCal object
        uvcal object containing fitted gains.
    fit_info:
        dictionary containing fit history for each time-step and polarization in the data with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpu_index is not None:
        # See https://www.tensorflow.org/guide/gpu
        if gpus:
            if gpu_memory_limit is None:
                tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            else:
                tf.config.set_logical_device_configuration(
                    gpus[gpu_index], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit * 1024)]
                )

            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    if isinstance(input_data_files, str):
        input_data_files = [input_data_files]
    if isinstance(input_data_files, list):
        uvd = UVData()
        uvd.read(input_data_files)
    else:
        uvd = input_data_files

    utils.select_baselines(uvd, bllen_min=bllen_min, bllen_max=bllen_max, bl_ew_min=bl_ew_min, ex_ants=ex_ants)

    if isinstance(input_model_files, str):
        input_model_files = [input_model_files]

    if input_model_files is not None:
        if isinstance(input_model_files, list):
            uvd_model = UVData()
            uvd_model.read(input_model_files)
        else:
            uvd_model = input_model_files
    else:
        uvd_model = None
    if uvd_model is not None:
        utils.select_baselines(uvd, bllen_min=bllen_min, bllen_max=bllen_max, bl_ew_min=bl_ew_min)

    if isinstance(input_gain_files, str):
        input_gain_files = [input_gain_files]
    if input_gain_files is not None:
        if isinstance(input_gain_files, list):
            uvc = UVCal()
            uvc.read_calfits(input_gain_files)
        else:
            uvc = input_gain_files
    else:
        uvc = None
    # run calibration with specified GPU device.
    if gpu_index is not None and gpus:
        with tf.device(f"/device:GPU:{gpus[gpu_index].name[-1]}"):
            model_fit, resid_fit, gains_fit, fit_info = calibrate_and_model_dpss(
                uvdata=uvd, sky_model=uvd_model, gains=uvc, **calibration_kwargs
            )
    else:
        model_fit, resid_fit, gains_fit, fit_info = calibrate_and_model_dpss(
            uvdata=uvd, sky_model=uvd_model, gains=uvc, **calibration_kwargs
        )

    if resid_outfilename is not None:
        resid_fit.write_uvh5(resid_outfilename, clobber=clobber)
    if gain_outfilename is not None:
        gains_fit.x_orientation = x_orientation
        gains_fit.write_calfits(gain_outfilename, clobber=clobber)
    if model_outfilename is not None:
        model_fit.write_uvh5(model_outfilename, clobber=clobber)
    # don't write fitting_info_outfilename for now.

    # don't write fitting_info_outfilename for now.
    return model_fit, resid_fit, gains_fit, fit_info


def input_output_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_argument_group("Input and Output Arguments.")
    sp.add_argument("--input_data_files", type=str, nargs="+", help="paths to data files to calibrate.", required=True)
    sp.add_argument(
        "--input_model_files", type=str, nargs="+", help="paths to model files to set overal amplitude and phase."
    )
    sp.add_argument("--input_gain_files", type=str, nargs="+", help="paths to gains to use as a staring point.")
    sp.add_argument("--resid_outfilename", type=str, default=None, help="postfix for resid output file.")
    sp.add_argument("--model_outfilename", type=str, default=None, help="postfix for foreground model file.")
    sp.add_argument("--gain_outfilename", type=str, default=None, help="path for writing fitted gains.")
    sp.add_argument("--clobber", action="store_true", default="False", help="Overwrite existing outputs.")
    sp.add_argument("--x_orientation", default="east", type=str, help="x_orientation of feeds to set in output gains.")
    sp.add_argument(
        "--bllen_min", default=0.0, type=float, help="minimum baseline length to include in calibration and outputs."
    )
    sp.add_argument(
        "--bllen_max", default=np.inf, type=float, help="maximum baseline length to include in calbration and outputs."
    )
    sp.add_argument(
        "--bl_ew_min",
        default=0.0,
        type=float,
        help="minimum EW baseline component to include in calibration and outputs.",
    )
    sp.add_argument(
        "--ex_ants", default=None, type=int, nargs="+", help="Antennas to exclude from calibration and modeling."
    )
    sp.add_argument("--gpu_index", default=None, type=int, help="Index of GPU to run on (if on a multi-GPU machine).")
    sp.add_argument("--gpu_memory_limit", default=None, type=int, help="Limit GPU memory use to this many GBytes.")
    return ap


def fitting_argparser():
    ap = input_output_parser()
    sp = ap.add_argument_group("General Fitting Arguments.")
    sp.add_argument(
        "--tol",
        type=float,
        default=1e-14,
        help="Stop gradient descent after cost function converges to within this value.",
    )
    sp.add_argument(
        "--optimizer", type=str, default="Adamax", help="First order optimizer to use for gradient descent."
    )
    sp.add_argument("--maxsteps", type=int, default=10000, help="Max number of steps to iterate during optimization.")
    sp.add_argument("--verbose", default=False, action="store_true", help="lots of text ouputs.")
    sp.add_argument(
        "--use_min",
        default=False,
        action="store_true",
        help="Use params for mimimum cost function derived. Otherwise, use the params last visited by the descent. Avoids momentum overshoot.",
    )
    sp.add_argument(
        "--use_redundancy",
        default=False,
        action="store_true",
        help="Model redundant visibilities with the same set of foreground parameters.",
    )
    sp.add_argument(
        "--correct_model", default=True, action="store_true", help="Remove gain effects from foreground model."
    )
    sp.add_argument(
        "--correct_resid", default=False, action="store_true", help="Apply fitted gains to the fitted residuals."
    )
    sp.add_argument(
        "--graph_mode",
        default=False,
        action="store_true",
        help="Pre-compile computational graph before running gradient descent. Not reccomended for GPUs.",
    )
    sp.add_argument(
        "--init_guesses_from_previous_time_step",
        default=False,
        action="store_true",
        help="initialize gain and foreground guesses from previous time step when calibrating multiple times.",
    )
    sp.add_argument("--learning_rate", type=float, default=1e-2, help="gradient descent learning rate.")
    sp.add_argument(
        "--red_tol", type=float, default=1.0, help="Tolerance for determining redundancy between baselines [meters]."
    )
    sp.add_argument(
        "--skip_threshold",
        type=float,
        default=0.5,
        help="Skip and flag time/polarization if more then this fractionf of data is flagged.",
    )
    sp.add_argument(
        "--model_regularization",
        type=str,
        default="post_hoc"
    )
    return ap


def dpss_fit_argparser():
    ap = fitting_argparser()
    sp = ap.add_argument_group("DPSS Specific Fitting Arguments.")
    sp.add_argument("--horizon", default=1.0, type=float, help="Fraction of horizon delay to model with DPSS modes.")
    sp.add_argument("--min_dly", default=0.0, type=float, help="Minimum delay [ns] to model with DPSS modes.")
    sp.add_argument(
        "--offset", default=0.0, type=float, help="Offset from horizon delay [ns] to model with DPSS modes."
    )
    return ap
