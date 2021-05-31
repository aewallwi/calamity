import numpy as np
from uvtools import dspec
import tensorflow as tf
from pyuvdata import UVData, UVCal
from . import utils

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def calibrate_data_model_per_baseline(uvdata, foreground_basis_vectors, fg0=None, g0=None, weights=None, pol=None, freeze_model=False,
                                      foreground_coefficients=None, optimizer='Adamax', use_redunancy=False, tol=1e-14, maxsteps=10000, **opt_kwargs):
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
    g0: UVCal object
        UVCal with initial gain estimates.
        There many smart ways to obtain initial gain estimates
        but this is beyond the scope of calamity (for example, firstcal, logcal, sky-based cal).
        Users can determine initial gains with their favorite established cal algorithm.
        default is None -> start with unity gains.
        WARNING: At the present, the flags in g0 are not propagated/used! Make sure flags in uvdata object!
    fg0: dict
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
    # use zeroth pol if pol is None.
    if pol is None:
        pol = uvdata.get_pols()[0]
    # pull data out of uvdata object and into a dict (to avoid slicing operations)
    data_dict = {}
    flag_dict = {}
    nsample_dict = {}
    gain_dict = {}
    for ap in uvdata.get_antpairs():
        bl = ap + (pol,)
        data_dict[ap] = uvdata.get_data(bl)
        flag_dict[ap] = uvdata.get_flags(bl)
        nsample_dict[ap] = uvdata.get_nsamples(bl)
    for ant in g0.antenna_numbers:
        gain_dict[ant] = g0.get_gains(ant, 'J' + pol)
    # initialize gains.
    if g0 is None:
        g0 = utils.blank_uvcal_from_uvdata()
    # initialize foreground modeling coefficients.
    if fg0 is None:
        fg0 = {ap: None for ap in foreground_basis_vectors}
        for ap in foreground_basis_vectors:
            data = uvdata.get_data()
            fg0[ap] = (data_dict[ap] / gain_dict[(ap[0], 'J' + pol) / np.conj(gain_dict[(ap[1], 'J' + pol)])) @ foreground_basis_vectors[ap] # gives Ntimes x Nbasis coefficients for each fg coeff.
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
        data_map_r = {(i, j): tf.convert_to_tensor(data_dict[(ant_map[i], ant_map[j])][tnum].real, dtype=np.float32) for i, j in itertools.combinations(ant_map.keys(), 2)}
        data_map_i = {(i, j): tf.convert_to_tensor(data_dict[(ant_map[i], ant_map[j])][tnum].imag, dtype=np.float32) for i, j in data_map_r.keys()}
        weights_map = {(i, j): tf.convert_to_tensor((~flag_dict[ap]).astype(np.float32)[tnum] * nsample_dict[ap][tnum], dtype=np.float32) for i, j in data_map_r.keys()}
        # initialize fg_coeffs to optimize
        fg = np.hstack([fg0[(ant_map[i], ant_map[j]))][tnum].squeeze()])
        fg_r = tf.Variable(tf.convert_to_tensor(fg.real, dtype=np.float32))
        fg_i = tf.Variable(tf.convert_to_tensor(fg.imag, dtype=np.float32))
        g_r = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].real for i in ant_map.keys()), dtype=np.float32))
        g_i = tf.Variable(tf.convert_to_tensor(np.vstack([gain_dict[ant_map[i]][tnum].imag for i in ant_map.keys()), dtype=np.float32))
        # define a loss function
        def loss:
            loss_total = 0.
            wsum = 0.
            for i, j in data_map.keys():
                # real part of loss
                loss_total += tf.reduce_sum(tf.square((g_r[i] * g_r[j] + g_i[i] * g_i[j]) * tf.reduce_sum(evec_map[i, j] * fg_r[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) \
                                                      (g_r[i] * g_i[j] - g_i[i] * g_r[j]) * tf.reduce_sum(evec_map[i, j] * fg_i[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) - data_map[i, j].real) * weights_map[i, j])
                # imag part of loss
                loss_total += tf.reduce_sum(tf.square((g_r[i] * g_r[j] + g_i[i] * g_i[j]) * tf.reduce_sum(evec_map[i, j] * fg_r[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) \
                                                      (g_i[i] * g_r[j] - g_r[i] * g_i[j]) * tf.reduce_sum(evec_map[i, j] * fg_i[fgrange_map[i, j][0]:fgrange_map[i, j][1]], axis=1) - data_map[i, j].imag) * weights_map[i, j])
                wsum += 2 * tf.reduce_sum(weights_map[i, j])
            return loss_total / wsum
        # initialize the optimizer.
        opt = OPTIMIZERS[optimizer](**opt_kwargs)
        # perform optimization loop.
