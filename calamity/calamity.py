import numpy as np
from uvtools import dspec
import tensorflow as tf
from pyuvdata import UVData, UVCal
from . import utils
import copy
import tqdm

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def calibrate_data_model_per_baseline(uvdata, foreground_basis_vectors, gains=None, freeze_model=False,
                                      foreground_coefficients=None, optimizer='Adamax', use_redunancy=False, tol=1e-14, maxsteps=10000,
                                      verbose=False, sky_model=None, **opt_kwargs):
    """Perform simultaneous calibration and fitting of foregrounds --per baseline--.

    This approach gives up on trying to invert the wedge but can be used on practically any array.

    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    foreground_basis_vectors: dictionary
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
    foreground_coeffs: dict
        Dictionary with baseline keys pointing to foreground coefficients for each baseline.
        default is None -> use dot product of model vectors with each data baseline to kick things off.
    weights: UVData object
        optional UVData object containing weights in the data-array.
        default is None -> use binary flag weights from uvdata flag array.
    optimizer: string
        Name of optimizer. See OPTIMIZERS dictionary
        default is 'Adamax'
    tol: float, optional
        halting condition for optimizer loop. Stop loop when the change in the cost function falls
        below tol.
        default is 1e-14
    maxsteps: int, optional
        maximum number of opt.minimize calls before halting.
        default is 10000

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
            'gr': real part of gains.
            'gi': imag part of gains
    """
    uvdata = uvdata.select(inplace=False, bls=[ap for ap in uvdata.get_antpairs() if ap[0] != ap[1]])
    resid = copy.deepcopy(uvdata)
    model = copy.deepcopy(uvdata)
    filtered = copy.deepcopy(uvdata)
    if polarizations is None:
        polarizations = uvdata.get_pols()
    if gains is None:
        gains = utils.blank_uvcal_from_uvdata()
    fitting_info = {}
    for pol, polnum in enumerate():
        fitting_info_p = {}
        rmsdata = np.sqrt(np.mean(np.abs(uvdata.data_array[:, :, :, polnum][~uvdata.flag_array][:, :, :, polnum]) ** 2.))
        # pull data out of uvdata object and into a dict (to avoid slicing operations)
        data_dict = {}
        model_dict = {}
        resid_dict = {}
        flag_dict = {}
        nsample_dict = {}
        gain_dict = {}
        for ap in uvdata.get_antpairs():
            bl = ap + (pol,)
            data_dict[ap] = uvdata.get_data(bl)
            model_dict[ap] = uvdata.get_data(bl)
            resid_dict[ap] = uvdata.get_data(bl)
            flag_dict[ap] = uvdata.get_flags(bl)
            nsample_dict[ap] = uvdata.get_nsamples(bl)
        for ant in gains.antenna_numbers:
            gain_dict[ant] = gains.get_gains(ant, 'J' + pol)
        # initialize foreground modeling coefficients.
        foreground_coeffs = {ap: None for ap in foreground_basis_vectors}
        for ap in foreground_basis_vectors:
            data = uvdata.get_data(ap + (pol,))
            foreground_coeffs[ap] = (data_dict[ap] / gain_dict[(ap[0], 'J' + pol) / np.conj(gain_dict[(ap[1], 'J' + pol)])) @ foreground_basis_vectors[ap] # gives Ntimes x Nbasis coefficients for each fg coeff.
        # perform solutions on each time separately.
        ant_map = {i: ant for i, ant in enumerate(g0.ant_array)}
        ant_mapi = {ant: i for ant, i in zip(ant_map.values(), ant_map.keys())}
        evec_map = {(i, j): foreground_basis_vectors[(ant_map[i], ant_map[j])] for i, j in itertools.combinations(ant_map.keys(), 2)}
        # map i, j to start-end indices of foreground vector.
        fgrange_map = {}
        startind = 0
        for i, j in evec_map.keys():
            fgrange_map[(i, j)] = (startind, startind + len(evec_map[(i, j)]))
            startind += len(evec_map[(i, j)])

        for tnum in uvdata.Ntimes:
            data_map_r = {}
            data_map_i = {}
            for i, j in itertools.combinations(ant_map.keys(), 2):
                data_map_r[(i, j)] = tf.convert_to_tensor(data_dict[(ant_map[i], ant_map[j])][tnum].real / rmsdata, dtype=np.float32)
                if (ant_map[i], ant_map[j]) in data_dict:
                    data_map_i[(i, j)] = tf.convert_to_tensor(data_dict[(ant_map[i], ant_map[j])][tnum].imag / rmsdata, dtype=np.float32)
                # store negative of imag (complex conj) if ant-pair is reversed.
                elif (ant_map[j], ant_map[i]) in data_dict:
                    data_map_i[(i, j)] = tf.convert_to_tensor(-data_dict[(ant_map[i], ant_map[j])][tnum].imag / rmsdata, dtype=np.float32)

            weights_map = {(i, j): tf.convert_to_tensor((~flag_dict[ap]).astype(np.float32)[tnum] * nsample_dict[ap][tnum], dtype=np.float32) for i, j in data_map_r.keys()}
            # initialize fg_coeffs to optimize
            fg = np.hstack([foreground_coeffs[(ant_map[i], ant_map[j]))][tnum].squeeze() / rmsdata])
            fg_r = tf.Variable(tf.convert_to_tensor(fg.real, dtype=np.float32))
            fg_i = tf.Variable(tf.convert_to_tensor(fg.imag, dtype=np.float32))
            g_r = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].real for i in ant_map.keys()), dtype=np.float32))
            g_i = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].imag for i in ant_map.keys()), dtype=np.float32))
            # get the foreground model.
            def yield_fg_model(i, j):
                vr = tf.reduce_sum(evec_map[i, j] * fg_r[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) # real part of fg model.
                vi = tf.reduce_sum(evec_map[i, j] * fg_i[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) # imag part of fg model.
                return vr, vi
            # get the calibrated model
            def yield_model(i, j):
                vr, vi = yield_fg_model(i, j)
                model_r = (g_r[i] * g_r[j] + g_i[i] * g_i[j]) *  vr + (g_r[i] * g_i[j] - g_i[i] * g_r[j]) * vi # real part of model with gains
                model_i = (g_r[i] * g_r[j] + g_i[i] * g_i[j]) * vi + (g_i[i] * g_r[j] - g_r[i] * g_i[j]) * vr # imag part of model with gains
                return model_r, model_i
            # function for computing loss to be minimized in optimization.
            def cal_loss():
                loss_total = 0.
                wsum = 0.
                for i, j in data_map_r.keys():
                    model_r, model_i = yield_model(i, j)
                    loss_total += tf.reduce_sum(tf.square(model_r  - data_map[i, j].real) * weights_map[i, j])
                    # imag part of loss
                    loss_total += tf.reduce_sum(tf.square(model_i - data_map[i, j].imag) * weights_map[i, j])
                    wsum += 2 * tf.reduce_sum(weights_map[i, j])
                return loss_total / wsum
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
            # perform optimization loop.
            for step in tqdm.tqdm(range(max_steps)):
                with tf.GradientTape() as tape:
                    loss = cal_loss()
                if freeze_model:
                    vars = [g_r, g_i]
                else:
                    vars = [g_r, g_i, fg_r, fg_i]
                grads = tape.gradient(loss, vars)
                opt.apply_gradients(zip(grads, vars))
                fitting_info_t['loss_history'].append(loss.numpy())
                if record_var_history:
                    fitting_info_t['g_r'].append(g_r.numpy())
                    fitting_info_t['g_i'].append(g_i.numpy())
                    if not freeze_model:
                        fitting_info_t['fg_r'].append(fg_r.numpy())
                        fitting_info_t['fg_i'].append(fg_i.numpy())

            ants_visited = set({})
            for ap in data_dict:
                i, j = ant_mapi[ap[0]], ant_mapi[ap[1]]
                model_r, model_i = calculate_model(i, j)
                model_fg_r, model_fg_i = calculate_fg_model(i, j)

                if ap[0] not in ants_visited:
                    gain_dict[ap[0]] = g_r[i].numpy() + 1j * g_i[i].numpy()
                    ants_visited.add(ap[0])
                if ap[1] not in ants_visited:
                    gain_dict[ap[1]] = g_r[j].numpy() + 1j * g_i[j].numpy()
                    ants_visited.add(ap[1])

                if i, j in data_map_r:
                    model = model_r.numpy() + 1j * model_i.numpy()
                    model_fg = model_fg_r.numpy() + 1j * model_fg_i.numpy()
                else:
                    model = model_r.numpy() - 1j * model_i.numpy()
                    model_fg = model_fg_r.numpy() + 1j * model_fg_i.numpy()

                # set foreground model. Flux scale and overall phase will be off.
                # we will fix this at the end.
                model_dict[ap][tnum] = model_fg * rmsdata
                # subtract model
                resid_dict[ap][tnum] -= model * rmsdata
                # divide by cal solution flux and phase scales will be off. We will fix this at the end.
                resid_dict[ap][tnum] = resid_dict[ap][tnum] / (gain_dict[ap[0]][tnum] * np.conj(gain_dict[ap[1]][tnum])
            fitting_info_p[tnum] = fitting_info_t

        # now fill in the resid data_array and the model data_array
        ants_visited = set({})
        for ap in resid_dict:
            dinds = resid.antpair2ind(ap)
            resid.data_array[dinds, 0, :, polnum] = resid_dict[ap]
            model.data_array[dinds, 0, :, polnum] = resid_dict[ap]
            model.flag_array[dinds, 0, :, polnum] = np.zeros_like(model.flag_array[dinds, 0, :, :, polnum])
            filtered.data_array[dinds, 0, :, polnum] = resid.data_array[dinds, 0, :, polnum] + model.data_array[dinds, 0, :, polnum]
            filtered.flag_array[dinds, 0, :, polnum] = model.flag_array[dinds, 0, :, polnum]
            if ap[0] not in ants_visited:
                gind = gains.antpair2ind(ap[0])
                gains.gain_array[gind, 0, :, :, polnum] = gain_dict[ap[0]].T
                ants_visited.add(ap[0])
            if ap[1] not in ants_visited:
                gind = gains.antpair2ind(ap[1])
                gains.gain_array[gind, 0, :, :, polnum] = gains_dict[ap[1]].T
        # free up memory
        del resid_dict, model_dict, data_dict, gain_dict
        # rescale by the abs rms of the data and an overall phase.
        if sky_model is None:
            sky_model = uvdata
        scale_factor_phase = np.angle(np.mean(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]])))
        scale_factor_abs = np.sqrt(np.mean(np.abs(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]]) ** 2.))
        scale_factor = scale_factor_abs * np.exp(1j * scale_factor_phase)
        model.data_array[:, :, :, polnum] *= scale_factor
        resid.data_array[:, :, :, polnum] *= scale_factor
        filtered.data_array[:, :, :, polnum] *= scale_factor
        gains.gain_array[:, :, :, polnum]  = gains.gain_array[:, :, :, polnum] / np.sqrt(scale_factor)

    return model, resid, filtered, gains, fitting_info


def calibrate_data_model_dpss(horizon=1., min_dly=0., offset=0., **fitting_kwargs):
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
    fitting_kwargs: kwarg dict
        additional kwargs for calibrate_data_model_per_baseline.
        see docstring of calibrate_data_model_per_baseline.

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
            'gr': real part of gains.
            'gi': imag part of gains
    """
    dpss_evecs = {}
    # generate dpss modeling vectors.
    for ap in uvdata.get_antpairs():
        i = np.argmin(np.abs(ap[0] - uvdata.ant_array))
        j = np.argmin(np.abs(ap[1] - uvdata.ant_array))
        dly = np.linalg.norm(uvdata.antenna_positions[i] - uvdata.antenna_positions[j]) / .3
        dly = max(min_dly, dly * horizon + offset) / 1e9
        dpss_evecs[ap] = dspec.dpss_operator(uvdata.freq_array[0], filter_centers=[0.0], filter_half_widths=[dly], eigenval_cutoff=[1e-12])
    model, resid, filtered, gains, fitted_info = calibrate_data_model_per_baseline(foreground_basis_vectors=dpss_evecs, **fitting_opts)


def read_calibrate_and_filter_data_per_baseline(infilename, resid_file=None, model_file=None, filtered_file=None, calfile=None, sky_model=None, modeling_basis='dpss'):
    """Driver function
    """

def red_calibrate_and_filter_data_argparser():
    """Get argparser for calibrating and filtering.
    """
