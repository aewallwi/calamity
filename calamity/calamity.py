import numpy as np
from uvtools import dspec
import tensorflow as tf
from pyuvdata import UVData, UVCal

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax':tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD':tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


def calibrate_data(uvdata, foreground_basis_vectors, fg0=None, g0=None, weights=None,
                   foreground_coefficients=None, optimizer='Adamax', tol=1e-14, maxsteps=10000, **opt_kwargs):
    """A foreground loss function

    Parameters
    ----------
    uvdata: UVData object
        uvdata objet of data to be calibrated.
    foreground_coefficients: array-like
        Nfg foreground coefficients
    foreground_basis_vectros: array-like
        (Nbls x Nfrequency) x Nfg 2d tensor of foreground basis vectors.
    fg0: array-like
        Nfg len complex 1d array of coefficients to serve as initial values for fg
        optimization.
    g0: array-like
        Nant x Nfreqs len complex 1d array of initial gain values.
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
