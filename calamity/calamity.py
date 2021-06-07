import numpy as np
import tensorflow as tf
from pyuvdata import UVData, UVCal
from . import utils
import copy
import argparse
import itertools
from .utils import echo
import datetime

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax': tf.optimizers.Adamax, 'AMSGrad': tf.optimizers.AMSGrad,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD': tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}




def tensorize_per_baseline_model_components(red_grps, foreground_modeling_components, dtype=dtype_opt, method='dictionary'):
    """Helper function generating mappings for per-baseline foreground modeling.

    Generates mappings between antenna pairs and foreground basis vectors accounting for redundancies.

    Parameters
    ----------
    red_grps: list of lists of int 2-tuples
        a list of lists of 2-tuples where all antenna pairs within each sublist
        are redundant with eachother. Assumes that conjugates are correctly taken.
    foreground_modeling_components: dict of 2-tuples as keys and numpy.ndarray as values.
        dictionary mapping int antenna-pair 2-tuples to

    Returns
    -------
    foreground_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.
    model_component_tensor_map: diction with 2-tuples as keys an tf.Tensor objects as values.
        dictionary mapping antenna pairs to
    """
    model_component_tensor_map = {}
    foreground_range_map = {}
    startind = 0
    for grpnum, red_grp in enumerate(red_grps):
        # set foreground_range_map value based on first antenna-pair in redundant group.
        ncomponents = foreground_modeling_components[red_grp[0]].shape[1]
        fg_range_map[red_grp[0]] = (startind, startind + ncomponents)
        startind += ncomponents
        for ap in red_grp:
            model_component_tensor_map[ap] = tf.convert_to_tensor(foreground_modeling_components[ap], dtype=dtype_opt)
            foreground_range_map[ap] = foreground_range_map[red_grp[0]]

    return foreground_range_map, model_component_tensor_map


def yield_foreground_model_per_baseline_dictionary_method(i, j, foreground_coeffs_real, foreground_coeffs_imag, foregroud_range_map, components_map):
    """Helper function for retrieving a per-baseline foreground model using the dictionary mapping technique

    From empirical experimentation, this technique works best in graph mode on CPUs. We recommend
    the array method if working with GPUs.

    Parameters
    ----------
    i: int
        i correlation index
    j: int
        j correlation index
    foreground_coeffs_real: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the real components of coefficients multiplying foreground
        basis vectors.
    foreground_coeffs_imag: tf.Tensor object
        an Nforegrounds tensorflow tensor
        storing the imaginary components of coefficients multiplying foreground
        basis vectors.
    foreground_range_map: dict with 2-tuple int keys and 2-int tuple values
        dictionary with keys that are (i, j) pairs of correlation indices which map to
        integer 2-tuples (index_low, index_high) representing the lower and upper indices of
        the foreground_coeffs tensor. Lower index is inclusive, upper index is exclusive
        (consistent with python indexing convention).
    components_map: dict of 2-tuple int keys and tf.Tensor object values.
        dictionary with keys that are (i, j) integer pairs of correlation indices which map to
        Nfreq x Nforeground tensorflow tensor where each column is a separate per-baseline
        foreground basis component.

    Returns
    -------
    foreground_model_real: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the real part of the (i, j) correlation.
    foreground_model_imag: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the imag part of the (i, j) correlation

    """
    foreground_model_real = tf.reduce_sum(components_map[i, j] * foreground_coeffs_real[foreground_range_map[(i, j)][0]: foreground_range_map[(i, j)][1]], axis=1) # real part of fg model.
    foreground_model_imag = tf.reduce_sum(components_map[i, j] * foreground_coeffs_imag[foreground_range_map[(i, j)][0]: foreground_range_map[(i, j)][1]], axis=1) # imag part of fg model.
    return foreground_model_real, foreground_model_imag


# get the calibrated model
def yield_data_model_per_baseline_dictionary(i, j, gains_real, gains_imag, **foreground_model_kwargs):
    """Helper function for retrieving a per-baseline uncalibrted foreground model using the dictionary mapping technique

    From empirical experimentation, this technique works best in graph mode on CPUs. We recommend
    the array method if working with GPUs.

    Parameters
    ----------
    i: int
        i correlation index
    j: int
        j correlation index
    gains_real: dict with int keys and tf.Tensor object values
        dictionary mapping i antenna numbers to Nfreq 1d tf.Tensor object
        representing the real component of the complex gain for antenna i.
    gains_imag: dict with int keys and tf.Tensor object values
        dictionary mapping j antenna numbers to Nfreq 1d tf.Tensor object
        representing the imag component of the complex gain for antenna j.
    **foreground_model_kwargs: kwargs dict
        kwargs for yield_foreground_model_per_baseline_dictionary_method (excluding i, and j)
        see yield_foreground_model_per_baseline_dictionary_method docstring.

    Returns
    -------
    uncal_model_real: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the real part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    uncal_model_imag: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the imag part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    """
    foreground_model_real, foreground_model_imag = yield_foreground_model_per_baseline_dictionary_method(i, j, **foreground_model_kwargs)
    uncal_model_real = (gains_real[i] * gains_real[j] + gains_imag[i] * gains_imag[j]) *  foreground_model_real + (gains_real[i] * gains_imag[j] - gains_imag[i] * gains_real[j]) * foreground_model_imag # real part of model with gains
    uncal_model_imag = (gains_real[i] * gains_real[j] + gains_imag[i] * gains_imag[j]) * foreground_model_imag + (gains_imag[i] * gains_real[j] - gains_real[i] * gains_imag[j]) * foreground_model_real # imag part of model with gains
    return uncal_model_real, uncal_model_imag


# tf.function decorator -> will pre-optimize computation in graph mode.
# lets us side-step hard-to-read and sometimes wasteful purely
# parallelpiped tensor computations. This leads to a x4 speedup
# in processing the data in test_calibrate_and_model_dpss
# on an i5 CPU macbook over pure python.
# TODO: See how this scales with array size.
# see https://www.tensorflow.org/guide/function.
@tf.function
def cal_loss_dictionary(gains_real, gains_imag, foreground_coeffs_real, foreground_coeffs_imag, data_real, data_imag, wgts, foreground_range_map):
    """MSE loss-function for dictionary method of computing data model.

    Parameters
    ----------
    gains_real: dictionary with ints as keys and tf.Tensor objects as values.
        dictionary mapping antenna numbers to Nfreq 1d tensors representing the
        real part of the model for each antenna.
    gains_imag: dictionary with ints as keys and tf.Tensor objects as values.
        dictionary mapping antenna numbers to Nfreq 1d tensors representing the
        imag part of the model for each antenna.
    foreground_coeffs_real: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to N-foreground-coeff 1d tensors representing the
        real part of the model for each antenna.
    foreground_coeffs_imag: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to N-foreground-coeff 1d tensors representing the
        imag part of the model for each antenna.
    data_real: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        the real part of the target data for each baseline.
    data_imag: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        the imag part of the target data for each baseline.
    wgts: dictionary with int 2-tuples as keys and tf.Tensor objects as values.
        dictionary mapping antenna-pairs to Nfreq 1d tensors containing
        per-frequency weights for each baseline contributing to the loss function.
    foreground_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.

    Returns
    -------
    loss_total: tf.Tensor scalar
        The MSE (mean-squared-error) loss value for the input model parameters and data.
    """
    loss_total = 0.
    for i, j in foreground_range_map:
        model_r, model_i = yield_data_model_per_baseline_dictionary(i, j, gains_real, gains_imag, foreground_coeffs_real, foreground_coeffs_imag)
        loss_total += tf.reduce_sum(tf.square(model_r  - data_real[i, j]) * wgts[i, j])
        # imag part of loss
        loss_total += tf.reduce_sum(tf.square(model_i - data_imag[i, j]) * wgts[i, j])
    return loss_total


def yield_foreground_model_per_baseline_tensor_method():
    return


def yield_data_model_per_baseline_tensor_method():
    return


def fit_data_dictionary_method(uvdata, time, polarization, red_grps):
    """

    """



def calibrate_and_model_per_baseline_dictionary_method(uvdata, foreground_modeling_components, gains=None, freeze_model=False,
                                                       optimizer='Adamax', tol=1e-14, maxsteps=10000, include_autos=False,
                                                       verbose=False, sky_model=None, dtype_opt=np.float32,
                                                       record_var_history=False, use_redundancy=False, notebook_progressbar=False,
                                                        break_if_loss_increases=True, **opt_kwargs):
    """Perform simultaneous calibration and fitting of foregrounds --per baseline--.

    This approach gives up on trying to invert the wedge but can be used on practically any array.

    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    foreground_modeling_components: dictionary
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
        a sky-model to use for initial estimates of foreground coefficients and
        to set overall flux scale and phases.
        Note that this model is not used to obtain initial gain estimates.
        These must be provided through the gains argument.
    dtype_opt: numpy dtype, optional
        the float precision to be used in tensorflow gradient descent.
        runtime scales roughly inversely linear with precision.
        default is np.float32
    record_var_history: bool, optional
        keep detailed record of optimization history of variables.
        default is False.
    use_redundancy: bool, optional
        if true, solve for one set of foreground coefficients per redundant baseline group
        instead of per baseline.
    notebook_progressbar: bool, optional
        use progress bar optimized for notebook output.
        default is False.
    break_if_loss_increases: bool, optional
        halt optimization loop if loss function increases.
        default is True.

    Returns
    -------
    uvdata_model: UVData object
        uvdata object containing model of the foregrounds multiplied by the gains
        (uncalibrated data). This model is mean to be subtracted from the data
        before applying gain solutions.
    uvcal_model: UVCal object
        uvcal object containing estimates of the gain solutions. These solutions
        are not referenced to any sky model and are likely orders of
    fitting_info:
        dictionary containing fit history with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coefficients
            'fg_i': imag part of foreground coefficients
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
        echo(f'{datetime.datetime.now()} Gains are None. Initializing gains starting with unity...\n', verbose=verbose)
        gains = utils.blank_uvcal_from_uvdata(uvdata)
    # if sky-model is None, initialize it to be the
    # data divided by the initial gain estimates.

    antpairs, red_grps, antpair_red_indices, _ = utils.get_redundant_groups_conjugated(uvdata, remove_redundancy=not(use_redundancy), include_autos=include_autos)

    if sky_model is None:
        echo(f'{datetime.datetime.now()} Sky model is None. Initializing from data...\n', verbose=verbose)
        sky_model = utils.apply_gains(uvdata, gains)

    fitting_info = {}
    echo(f'{datetime.datetime.now()} Generating map between antenna pairs and modeling vectors...\n', verbose=verbose)
    foreground_range_map, model_components_map = tensorize_per_baseline_model_components(red_grps, foreground_modeling_components, dtype=dtype_opt, method='dictionary')
    # We do fitting per time and per polarization and time.
    for polnum, pol in enumerate(uvdata.get_pols()):
        echo(f'{datetime.datetime.now()} Working on pol {pol}, {polnum + 1} of {uvdata.Npols}...\n', verbose=verbose)
        fitting_info_p = {}
        for tnum in range(uvdata.Ntimes):
            echo(f'{datetime.datetime.now()} Working on time {tnum + 1} of {uvdata.Ntimes}...\n', verbose=verbose)
            # pull data for pol out of raveled uvdata object and into dicts of 1d tf.Tensor objects for processing..
            echo(f'{datetime.datetime.now()} Tensorizing Data...\n', verbose=verbose)
            data_r, data_i, wgts = tensorize_per_baseline_data(uvdata, method='dictionary', dtype=dtype_opt, tnum=tnum, polnum=polnum)
            echo(f'{datetime.datetime.now()} Tensorizing Gains...\n', verbose=verbose)
            gain_r, gain_i = tensorize_gains(gains, method='dictionary', dtype=dtype_opt, tnum=tnum, polnum=polnum)
            echo(f'{datetime.datetime.now()} Tensorizing Foreground Coefficients...\n', verbose=verbose)
            fg_r, fg_i = tensorize_foreground_coeffs(sky_model, red_grps, dtype=dtype_opt, tnum=tnum, polnum=polnum)
            gain_r = tf.Variable(gain_r)
            gain_i = tf.Variable(gain_i)
            if not freeze_model:
                fg_r = tf.Variable(fg_r)
                fg_i = tf.Variable(fg_i)
            # initialize the optimizer.
            opt = OPTIMIZERS[optimizer](**opt_kwargs)
            # set up history recording
            fitting_info_t = {'loss_history':[]}
            if record_var_history:
                fitting_info_t['g_r'] = []
                fitting_info_t['g_i'] = []
                if not freeze_model:
                    fitting_info_t['fg_r'] = []
                    fitting_info_t['fg_i'] = []
            echo(f'{datetime.datetime.now()} Building Computational Graph...\n', verbose=(verbose and tnum == 0 and polnum == 0))
            # evaluate loss once to build graph.
            cal_loss_dictionary(g_r, g_i, fg_r, fg_i, data_r, data_i, wgts, foreground_range_map)
            echo(f'{datetime.datetime.now()} Performing Gradient Descent...\n', verbose=verbose)
            # perform optimization loop.
            if freeze_model:
                vars = [g_r, g_i]
            else:
                vars = [g_r, g_i, fg_r, fg_i]
            if notebook_progressbar:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            for step in tqdm(range(maxsteps)):
                with tf.GradientTape() as tape:
                    loss = cal_loss_dictionary(g_r, g_i, fg_r, fg_i, data_r, data_i, wgts, foreground_range_map)
                grads = tape.gradient(loss, vars)
                opt.apply_gradients(zip(grads, vars))
                fitting_info_t['loss_history'].append(loss.numpy())
                if record_var_history:
                    fitting_info_t['g_r'].append(g_r.numpy())
                    fitting_info_t['g_i'].append(g_i.numpy())
                    if not freeze_model:
                        fitting_info_t['fg_r'].append(fg_r.numpy())
                        fitting_info_t['fg_i'].append(fg_i.numpy())
                if step >= 1 and np.abs(fitting_info_t['loss_history'][-1] - fitting_info_t['loss_history'][-2]) < tol:
                    break
                # insert model values.
                



        # initialize foreground modeling coefficients.
        # these coefficients are mapped from (a1, a2) -> redundant_index -> fg_model_components
        foreground_coeffs = {}
        # model_dict follows (a1, a2) -> red_index -> model waterfall.
        model_dict = {}
        for red_grp in red_grps:
            model_dict[antpair_red_index[red_grp[0]]] = copy.copy(sky_model.get_data(red_grp[0] + (pol,)))
            foreground_coeffs[antpair_red_index[red_grp[0]]] =  model_dict[antpair_red_index[red_grp[0]]] @ foreground_modeling_components[red_grp[0]].astype(np.complex128)

        # perform solutions on each time separately.
        for tnum in range(uvdata.Ntimes):
            data_map_r = {}
            data_map_i = {}
            weights_map = {}
            fg_r = []
            fg_i = []
            # map data and gains from particular time into correlation indexed
            # tensor arrays to be used in optimization.
            echo(f'{datetime.datetime.now()} Building corr index maps...\n', verbose=verbose)
            for i, j in corrinds:
                ap = (ant_map[i], ant_map[j])
                isign = 1.
                # Use conjugation consistent with i conj(j).
                if ap not in antpairs:
                    ap = ap[::-1]
                    isign = -1.
                data_map_r[(i, j)] = tf.convert_to_tensor(data_dict[ap][tnum].real / rmsdata, dtype=dtype_opt)
                data_map_i[(i, j)] = tf.convert_to_tensor(isign * data_dict[ap][tnum].imag / rmsdata, dtype=dtype_opt)
                weights_map[(i, j)] = tf.convert_to_tensor((~flag_dict[ap]).astype(dtype_opt)[tnum] * nsample_dict[ap][tnum], dtype=dtype_opt)

            # order the foreground_coeffs by redundant group.
            # make sure to use i, conj(j) conjugation convention for foreground coefficients.
            for ap_index in range(len(red_grps)):
                fg_r.append(foreground_coeffs[ap_index][tnum].real.squeeze() / rmsdata)
                ap = red_grps[ap_index][0]
                if (ant_mapi[ap[0]], ant_mapi[ap[1]]) in corrinds:
                    fg_i.append(foreground_coeffs[ap_index][tnum].imag.squeeze() / rmsdata)
                else:
                    fg_i.append(-foreground_coeffs[ap_index][tnum].imag.squeeze() / rmsdata)


            # initialize fg_coeffs to optimize
            fg_r = tf.Variable(tf.convert_to_tensor(np.hstack(fg_r), dtype=dtype_opt))
            fg_i = tf.Variable(tf.convert_to_tensor(np.hstack(fg_i), dtype=dtype_opt))
            g_r = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].real for i in ant_map.keys()]), dtype=dtype_opt))
            g_i = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].imag for i in ant_map.keys()]), dtype=dtype_opt))
            # get the foreground model.
            def yield_fg_model(i, j):
                vr = tf.reduce_sum(evec_map[i, j] * fg_r[fgrange_map[(i, j)][0]: fgrange_map[(i, j)][1]], axis=1) # real part of fg model.
                vi = tf.reduce_sum(evec_map[i, j] * fg_i[fgrange_map[(i, j)][0]: fgrange_map[(i, j)][1]], axis=1) # imag part of fg model.
                return vr, vi
            # get the calibrated model
            def yield_model(i, j):
                vr, vi = yield_fg_model(i, j)
                model_r = (g_r[i] * g_r[j] + g_i[i] * g_i[j]) *  vr + (g_r[i] * g_i[j] - g_i[i] * g_r[j]) * vi # real part of model with gains
                model_i = (g_r[i] * g_r[j] + g_i[i] * g_i[j]) * vi + (g_i[i] * g_r[j] - g_r[i] * g_i[j]) * vr # imag part of model with gains
                return model_r, model_i

            @tf.function
            def cal_loss():
                loss_total = 0.
                wsum = 0.
                for i, j in corrinds:
                    model_r, model_i = yield_model(i, j)
                    loss_total += tf.reduce_sum(tf.square(model_r  - data_map_r[i, j]) * weights_map[i, j])
                    # imag part of loss
                    loss_total += tf.reduce_sum(tf.square(model_i - data_map_i[i, j]) * weights_map[i, j])
                    wsum += 2 * tf.reduce_sum(weights_map[i, j])
                return loss_total / wsum

                if break_if_loss_increases and step >= 1 and fitting_info_t['loss_history'][-1] > fitting_info_t['loss_history'][-2]:
                    break

            echo(f"\n{datetime.datetime.now()} Finished Optimization. Delta Loss is {np.abs(fitting_info_t['loss_history'][-1] - fitting_info_t['loss_history'][-2]):.2e}.", verbose=verbose)
            echo(f"Initial loss was {fitting_info_t['loss_history'][0]}. Final Loss is {fitting_info_t['loss_history'][-1]}.", verbose=verbose)
            echo(f"Transferring fitted model and gains to dictionaries...\n", verbose=verbose)
            ants_visited = set({})
            red_grps_visited = set({})
            # insert calibrated / model subtracted data / gains
            # into antenna keyed dictionaries.
            for ap in data_dict:
                i, j = ant_mapi[ap[0]], ant_mapi[ap[1]]
                apg = (ap[0], ap[1])
                if (i, j) not in corrinds:
                    i, j = j, i
                    isign = -1.
                    apg = apg[::-1]
                model_r, model_i = yield_model(i, j)
                model_fg_r, model_fg_i = yield_fg_model(i, j)

                if apg[0] not in ants_visited:
                    gain_dict[apg[0]] = g_r[i].numpy() + 1j * g_i[i].numpy()
                    ants_visited.add(apg[0])
                if apg[1] not in ants_visited:
                    gain_dict[apg[1]] = g_r[j].numpy() + 1j * g_i[j].numpy()
                    ants_visited.add(apg[1])

                model_cal = model_r.numpy() + isign * 1j * model_i.numpy()
                model_fg = model_fg_r.numpy() + isign * 1j * model_fg_i.numpy()


                # set foreground model. Flux scale and overall phase will be off.
                # we will fix this at the end.
                if antpair_red_index[ap] not in red_grps_visited:
                    model_dict[antpair_red_index[ap]][tnum] = model_fg * rmsdata
                    red_grps_visited.add(antpair_red_index[ap])
                # subtract model
                resid_dict[ap][tnum] -= model_cal * rmsdata
                # divide by cal solution flux and phase scales will be off. We will fix this at the end.
                resid_dict[ap][tnum] = resid_dict[ap][tnum] / (gain_dict[ap[0]][tnum] * np.conj(gain_dict[ap[1]][tnum]))

            fitting_info_p[tnum] = fitting_info_t

        # now transfer antenna keyed dictionaries to uvdata/uvcal objects for
        # i/o
        ants_visited = set({})
        echo(f'{datetime.datetime.now()} Raveling data from polarization {pol}...\n')
        for ap in antpairs_data:
            dinds = resid.antpair2ind(ap)
            if ap in resid_dict:
                resid.data_array[dinds, 0, :, polnum] = resid_dict[ap]
                model.data_array[dinds, 0, :, polnum] = model_dict[antpair_red_index[ap]]
            else:
                resid.data_array[dinds, 0, :, polnum] = np.conj(resid_dict[ap[::-1]])
                model.data_array[dinds, 0, :, polnum] = np.conj(model_dict[antpair_red_index[ap[::-1]]])

            model.flag_array[dinds, 0, :, polnum] = np.zeros_like(model.flag_array[dinds, 0, :, polnum])
            filtered.data_array[dinds, 0, :, polnum] = resid.data_array[dinds, 0, :, polnum] + model.data_array[dinds, 0, :, polnum]
            filtered.flag_array[dinds, 0, :, polnum] = model.flag_array[dinds, 0, :, polnum]
            if ap[0] not in ants_visited:
                gind = gains.ant2ind(ap[0])
                gains.gain_array[gind, 0, :, :, polnum] = gain_dict[ap[0]].T
                ants_visited.add(ap[0])
            if ap[1] not in ants_visited:
                gind = gains.ant2ind(ap[1])
                gains.gain_array[gind, 0, :, :, polnum] = gain_dict[ap[1]].T

        # free up memory
        del resid_dict, model_dict, data_dict, gain_dict
        # rescale by the abs rms of the data and an overall phase.
        scale_factor_phase = np.angle(np.mean(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]]))
        scale_factor_abs = np.sqrt(np.mean(np.abs(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]]) ** 2.))
        scale_factor = scale_factor_abs * np.exp(1j * scale_factor_phase)
        model.data_array[:, :, :, polnum] *= scale_factor
        resid.data_array[:, :, :, polnum] *= scale_factor
        filtered.data_array[:, :, :, polnum] *= scale_factor
        gains.gain_array[:, :, :, polnum]  = gains.gain_array[:, :, :, polnum] / np.sqrt(scale_factor)
        fitting_info[polnum] = fitting_info_p

    return model, resid, filtered, gains, fitting_info


def calibrate_and_model_dpss(uvdata, horizon=1., min_dly=0., offset=0., include_autos=False, verbose=False, **fitting_kwargs):
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
        additional kwargs for calibrate_and_model_per_baseline.
        see docstring of calibrate_and_model_per_baseline.

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
    fitting_info:
        dictionary containing fit history for each time-step and polarization in the data with fields:
        'loss_history': list of values of the loss function in each minimization iteration.
        if record_var_history is True:
            each of the following fields is included which points to an Nstep x Nvar array of values
            'fg_r': real part of foreground coefficients
            'fg_i': imag part of foreground coefficients
            'g_r': real part of gains.
            'g_i': imag part of gains
    """
    dpss_model_components = utils.yield_dpss_model_components(uvdata, horizon=horizon, min_dly=min_dly, offset=offset, include_autos=include_autos)
    model, resid, filtered, gains, fitted_info = calibrate_and_model_per_baseline(uvdata=uvdata, foreground_modeling_components=dpss_model_components,
                                                                                  include_autos=include_autos, verbose=verbose, **fitting_kwargs)
    return model, resid, filtered, gains, fitted_info


def read_calibrate_and_model_per_baseline(infilename, incalfilename=None, refmodelname=None, residfilename=None,
                                          modelfilename=None, filteredfilename=None, calfilename=None, modeling_basis='dpss',
                                          clobber=False, **cal_kwargs):
    """Driver

    Parameters
    ----------
    infilename: str
        path to the input uvh5 data file with data to calibrate and filter.
    incalefilename: str, optional
        path to input calfits calibration file to use as a starting point for gain solutions.
    refmodelname: str, optional
        path to an optional reference sky model that can be used to set initial gains and foreground coefficients.
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
        kwargs for calibrate_data_model_dpss and calibrate_and_model_per_baseline
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
        sky_model=None
    if modeling_basis == 'dpss':
        model, resid, filtered, gains, fitted_info = calibrate_and_model_dpss(uvdata=uvdata, sky_model=sky_model, gains=gains,
                                                                              **cal_kwargs)
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
        parser for running read_calibrate_and_filter_data_per_baseline with modeling_basis='dpss'

    """
    ap = argparse.ArgumentParser(description="Simultaneous Gain Calibration and Filtering of Foregrounds using DPSS modes")
    io_opts = ap.add_argument_group(title="I/O options.")
    io_opts.add_argument("infilename", type=str, help="Path to data file to be calibrated and modeled.")
    io_opts.add_argument("--incalfilename", type=str, help="Path to optional initial gain files.", default=None)
    io_opts.add_argument("--refmodelname", type=str, help="Path to a reference sky model that can be used to initialize foreground coefficients and set overall flux scale and phase.")
    io_opts.add_argument("--residfilename", type=str, help="Path to write output uvh5 residual.", default=None)
    io_opts.add_argument("--modelfilename", type=str, help="Path to write output uvh5 model.", default=None)
    io_opts.add_argument("--filteredfilename", type=str, help="Path to write output uvh5 filtered and calibrated data.")
    io_opts.add_argument("--calfilename", type=str, help="path to write output calibration gains.")
    fg_opts = ap.add_argument_group(title="Options for foreground modeling.")
    fg_opts.add_argument("--horizon", type=float, default=1.0, help="Fraction of horizon delay to model with DPSS modes.")
    fg_opts.add_argument("--offset", type=float, default=0.0, help="Offset off of horizon delay (in ns) to model foregrounds with DPSS modes.")
    fg_opts.add_argument("--min_dly", type=float, default=0.0, help="minimum delay, regardless of baseline length, to model foregrounds with DPSS modes.")
    fit_opts = ap.add_argument_group(title="Options for fitting and optimization")
    fit_opts.add_argument("--freeze_model", default=False, action="store_true", help="Only optimize gains (freeze foreground model on existing data ore user provided sky-model file).")
    fit_opts.add_argument("--optimizer", default="Adamax", type=str, help="Optimizer to use in gradient descent. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers")
    fit_opts.add_argument("--tol", type=float, default=1e-14, help="Halt optimization if loss (chsq / ndata) changes by less then this amount.")
    fit_opts.add_argument("--maxsteps", type=int, default=10000, help="Maximum number of steps to carry out optimization to")
    fit_opts.add_argument("--verbose", default=False, action="store_true", help="Lots of outputs.")
    fit_opts.add_argument("--learning_rate", default=1e-2, type=float, help="Initial learning rate for optimization")
    return ap
