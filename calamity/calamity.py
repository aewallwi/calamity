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
    # initialize gains.
    if g0 is None:
        g0 = utils.blank_uvcal_from_uvdata()
