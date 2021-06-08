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

OPTIMIZERS = {'Adadelta': tf.optimizers.Adadelta, 'Adam': tf.optimizers.Adam, 'Adamax': tf.optimizers.Adamax,
              'Ftrl': tf.optimizers.Ftrl, 'Nadam':tf.optimizers.Nadam, 'SGD': tf.optimizers.SGD, 'RMSprop': tf.optimizers.RMSprop}


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
    gains_real: tf.Tensor object.
        tensor object holding real component of gains
        for time_index and polarization
        shape is Nant x Nfreq
    gains_imag: tf.Tensor object.
        tensor object holding imag component of gains
        for time_index and polarization
        shape is Nant x Nfreq

    """
    polnum = np.where(uvcal.jones_array == uvutils.polstr2num(polarization, x_orientation=uvcal.x_orientation))[0][0]
    gains_real = tf.convert_to_tensor(uvcal.gain_array[:, 0, :, time_index, polnum].squeeze().real, dtype=dtype)
    gains_imag = tf.convert_to_tensor(uvcal.gain_array[:, 0, :, time_index, polnum].squeeze().imag, dtype=dtype)
    return gains_real, gains_imag

def tensorize_per_baseline_model_components_dictionary(red_grps, foreground_modeling_components, dtype=np.float32):
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
        foreground_range_map[red_grp[0]] = (startind, startind + ncomponents)
        startind += ncomponents
        for ap in red_grp:
            model_component_tensor_map[ap] = tf.convert_to_tensor(foreground_modeling_components[ap], dtype=dtype)
            foreground_range_map[ap] = foreground_range_map[red_grp[0]]
    return foreground_range_map, model_component_tensor_map


def yield_foreground_model_per_baseline_dictionary_method(i, j, foreground_coeffs_real, foreground_coeffs_imag, foreground_range_map, components_map):
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


def insert_model_into_uvdata_dictionary(uvdata, time_index, polarization, foreground_range_map, model_components_map, foreground_coeffs_real, foreground_coeffs_imag, scale_factor=1.0):
    """Insert tensor values back into uvdata object.

    Parameters
    ----------
    uvdata: UVData object
        uvdata object to insert model data into.
    time_index: int
        time index to insert model data at.
    polarization: str
        polarization to insert.
    foreground_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline model component number.
    components_map: dict of 2-tuple int keys and tf.Tensor object values.
        dictionary with keys that are (i, j) integer pairs of correlation indices which map to
        Nfreq x Nforeground tensorflow tensor where each column is a separate per-baseline
        foreground basis component.
    foreground_coeffs_real: tf.Tensor object
        1d tensor containing real parts of coefficients for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    foreground_coeffs_imag: tf.Tensor object
        1d tensor containing imag parts of coefficients for each modeling vector.
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
    polnum = np.where(uvdata.polarization_array == uvutils.polstr2num(polarization, x_orientation=uvdata.x_orientation))[0][0]
    for ap in foreground_range_map:
        fgrange = slice(foreground_range_map[ap][0], foreground_range_map[ap][1])
        model = model_components_map[ap].numpy() @ (foreground_coeffs_real.numpy()[fgrange] + 1j * foreground_coeffs_imag.numpy()[fgrange])
        model *= scale_factor
        if ap in data_antpairs:
            dinds = uvdata.antpair2ind(ap)[time_index]
        else:
            dinds = uvdata.antpair2ind(ap[::-1])[time_index]
            model = np.conj(model)
        uvdata.data_array[dinds, 0, :, polnum] = model


def insert_gains_into_uvcal_dictionary(uvcal, time_index, polarization, gains_real, gains_imag, ants_map):
    """Insert tensorized gains back into uvcal object

    Parameters
    ----------
    uvdata: UVData object
        uvdata object to insert model data into.
    time_index: int
        time index to insert model data at.
    polarization: str
        polarization to insert.
    gains_real: dict with int keys and tf.Tensor object values
        dictionary mapping i antenna numbers to Nfreq 1d tf.Tensor object
        representing the real component of the complex gain for antenna i.
    gains_imag: dict with int keys and tf.Tensor object values
        dictionary mapping j antenna numbers to Nfreq 1d tf.Tensor object
        representing the imag component of the complex gain for antenna j.
    ants_map: dict mapping integer keys to integer values.
        dictionary mapping antenna number to antenna index in gain array to antenna number.

    Returns
    -------
    N/A: Modifies uvcal inplace.
    """
    polnum = np.where(uvcal.jones_array == uvutils.polstr2num(polarization, x_orientation=uvcal.x_orientation))[0][0]
    for ant in ants_map:
        uvcal.gain_array[ants_map[ant], 0, :, time_index, polnum] == gains_real[ants_map[ant]].numpy() + 1j * gains_imag[ants_map[ant]].numpy()


# get the calibrated model
def yield_data_model_per_baseline_dictionary(i, j, gains_real, gains_imag, ants_map, foreground_coeffs_real, foreground_coeffs_imag, foreground_range_map, components_map):
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
    ants_map: dict mapping integer keys to integer values.
        dictionary mapping antenna number to antenna index in gains_real and gains_imag.
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
    uncal_model_real: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the real part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    uncal_model_imag: tf.Tensor object
        Nfreq 1d tensorflow tensor model of the imag part of the uncalibrated (i, j) correlation
        Real(V_{ij}^{true} \times g_i \times conj(g_j))
    """
    foreground_model_real, foreground_model_imag = yield_foreground_model_per_baseline_dictionary_method(i, j, foreground_coeffs_real=foreground_coeffs_real,
                                                                                                         foreground_coeffs_imag=foreground_coeffs_imag,
                                                                                                         foreground_range_map=foreground_range_map, components_map=components_map)
    i, j = ants_map[i], ants_map[j]
    uncal_model_real = (gains_real[i] * gains_real[j] + gains_imag[i] * gains_imag[j]) *  foreground_model_real + (gains_real[i] * gains_imag[j] - gains_imag[i] * gains_real[j]) * foreground_model_imag # real part of model with gains
    uncal_model_imag = (gains_real[i] * gains_real[j] + gains_imag[i] * gains_imag[j]) * foreground_model_imag + (gains_imag[i] * gains_real[j] - gains_real[i] * gains_imag[j]) * foreground_model_real # imag part of model with gains
    return uncal_model_real, uncal_model_imag


def cal_loss_dictionary(gains_real, gains_imag, foreground_coeffs_real, foreground_coeffs_imag, data_real, data_imag, wgts, ants_map, foreground_range_map, components_map):
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
    ants_map: dict mapping integer keys to integer values.
        dictionary mapping antenna number to antenna index in gains_real and gains_imag.
    foreground_range_map: dict with int 2-tuples as keys and int 2-tuples as values.
        dictionary mapping antenna pairs to index ranges of a foreground coefficient vector raveled by baseline and baseline eigenvector number.

    Returns
    -------
    loss_total: tf.Tensor scalar
        The MSE (mean-squared-error) loss value for the input model parameters and data.
    """
    loss_total = 0.
    for i, j in foreground_range_map:
        model_r, model_i = yield_data_model_per_baseline_dictionary(i, j, gains_real=gains_real, gains_imag=gains_imag, ants_map=ants_map,
                                                                    foreground_coeffs_real=foreground_coeffs_real, foreground_coeffs_imag=foreground_coeffs_imag,
                                                                    foreground_range_map=foreground_range_map, components_map=components_map)
        loss_total += tf.reduce_sum(tf.square(model_r  - data_real[i, j]) * wgts[i, j])
        # imag part of loss
        loss_total += tf.reduce_sum(tf.square(model_i - data_imag[i, j]) * wgts[i, j])
    return loss_total


def tensorize_per_baseline_data_dictionary(uvdata, polarization, time_index, red_grps, scale_factor=1.0,
                                           wgts_scale_factor=1.0, dtype=np.float32):
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
    data_real: dict with int 2-tuple keys and tf.Tensor values
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each
        representing the real part of the spectum for baseline (i, j) with pol
        polarization at time-index time_index.
    data_imag: dict with int 2-tuple keys and tf.Tensor values
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each
        representing the imag part of the spectum for baseline (i, j) with pol
        polarization at time-index time_index.
    wgts: dict with int 2-tuple keys and tf.Tensor values.
        dictionary mapping antenna 2-tuples to Nfreq tf.Tensor objects with each representing
        the real weigths

    """
    data_real = {}
    data_imag = {}
    wgts = {}
    for red_grp in red_grps:
        for ap in red_grp:
            bl = ap + (polarization, )
            data_real[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].real / scale_factor, dtype=dtype)
            data_imag[ap] = tf.convert_to_tensor(uvdata.get_data(bl)[time_index].imag / scale_factor, dtype=dtype)
            wgts[ap] = tf.convert_to_tensor(~uvdata.get_flags(bl)[time_index] * uvdata.get_nsamples(bl)[time_index] / wgts_scale_factor, dtype=dtype)
    return data_real, data_imag, wgts


def tensorize_foreground_coeffs(uvdata, modeling_component_dict, red_grps, time_index, polarization, scale_factor=1.0, dtype=np.float32):
    """Initialize foreground coefficient tensors from uvdata and modeling component dictionaries.


    Parameters
    ----------
    uvdata: UVData object.
        UVData object holding model data.
    modeling_component_dict: dict with int 2-tuple keys and tf.Tensor values.
        dictionary holding int 2-tuple keys mapping to Nfreq x Nfg tf.Tensors
        used to model each individual baseline.
    red_grps: list of lists of int 2-tuples
        lists of redundant baseline groups with antenna pairs set to avoid conjugation.
    time_index: int
        time index of data to calculate foreground coefficients for.
    polarization: str
        polarization to calculate foreground coefficients for.
    scale_factor: float, optional
        factor to scale data by.
        default is 1.
    dtype: numpy.dtype
        data type to store tensors.

    Returns
    -------
    foreground_coeffs_real: tf.Tensor object
        1d tensor containing real parts of coefficients for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    foreground_coeffs_imag: tf.Tensor object
        1d tensor containing imag parts of coefficients for each modeling vector.
        ordering is over foreground modeling vector per redundant group and then
        redundant group in the order of groups appearing in red_grps
    """
    foreground_coefficients_real = []
    foreground_coefficients_imag = []
    for red_grp in red_grps:
        ap = red_grp[0]
        bl = ap + (polarization, )
        foreground_coefficients_real.extend(uvdata.get_data(bl).real[time_index] / scale_factor * ~uvdata.get_flags(bl)[time_index] \
                                            @ modeling_component_dict[ap].numpy())
        foreground_coefficients_imag.extend(uvdata.get_data(bl).imag[time_index] / scale_factor * ~uvdata.get_flags(bl)[time_index] \
                                            @ modeling_component_dict[ap].numpy())
    foreground_coefficients_real = tf.convert_to_tensor(np.asarray(foreground_coefficients_real), dtype=dtype)
    foreground_coefficients_imag = tf.convert_to_tensor(np.asarray(foreground_coefficients_imag), dtype=dtype)

    return foreground_coefficients_real, foreground_coefficients_imag


def calibrate_and_model_per_baseline_dictionary_method(uvdata, foreground_modeling_components, gains=None, freeze_model=False,
                                                       optimizer='Adamax', tol=1e-14, maxsteps=10000, include_autos=False,
                                                       verbose=False, sky_model=None, dtype=np.float32,
                                                       record_var_history=False, use_redundancy=False, notebook_progressbar=False,
                                                       **opt_kwargs):
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
    dtype: numpy dtype, optional
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
    foreground_range_map, model_components_map = tensorize_per_baseline_model_components_dictionary(red_grps, foreground_modeling_components, dtype=dtype)
    # index antenna numbers (in gain vectors).
    ants_map = {gains.antenna_numbers[i]: i for i in range(len(gains.antenna_numbers))}
    # We do fitting per time and per polarization and time.
    for polnum, pol in enumerate(uvdata.get_pols()):
        echo(f'{datetime.datetime.now()} Working on pol {pol}, {polnum + 1} of {uvdata.Npols}...\n', verbose=verbose)
        fitting_info_p = {}
        for time_index in range(uvdata.Ntimes):
            rmsdata = np.sqrt(np.mean(np.abs(uvdata.data_array[time_index ::uvdata.Ntimes, 0, :, polnum][~uvdata.flag_array[time_index ::uvdata.Ntimes, 0, :, polnum]]) ** 2.))
            wgtsum = np.sum(uvdata.nsample_array[time_index:: uvdata.Ntimes, 0, :, polnum] * ~uvdata.flag_array[time_index:: uvdata.Ntimes, 0, :, polnum])
            echo(f'{datetime.datetime.now()} Working on time {time_index + 1} of {uvdata.Ntimes}...\n', verbose=verbose)
            # pull data for pol out of raveled uvdata object and into dicts of 1d tf.Tensor objects for processing..
            echo(f'{datetime.datetime.now()} Tensorizing Data...\n', verbose=verbose)
            data_r, data_i, wgts = tensorize_per_baseline_data_dictionary(uvdata, dtype=dtype, time_index=time_index, polarization=pol, scale_factor=rmsdata, wgts_scale_factor=wgtsum, red_grps=red_grps)
            echo(f'{datetime.datetime.now()} Tensorizing Gains...\n', verbose=verbose)
            gain_r, gain_i = tensorize_gains(gains, dtype=dtype, time_index=time_index, polarization=pol)
            echo(f'{datetime.datetime.now()} Tensorizing Foreground Coefficients...\n', verbose=verbose)
            fg_r, fg_i = tensorize_foreground_coeffs(uvdata=sky_model, red_grps=red_grps, modeling_component_dict=model_components_map,
                                                     dtype=dtype, time_index=time_index, polarization=pol, scale_factor=rmsdata)
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
            echo(f'{datetime.datetime.now()} Building Computational Graph...\n', verbose=(verbose and time_index == 0 and polnum == 0))
            # evaluate loss once to build graph.

            # tf.function decorator -> will pre-optimize computation in graph mode.
            # lets us side-step hard-to-read and sometimes wasteful purely
            # parallelpiped tensor computations. This leads to a x4 speedup
            # in processing the data in test_calibrate_and_model_dpss
            # on an i5 CPU macbook over pure python.
            # TODO: See how this scales with array size.
            # see https://www.tensorflow.org/guide/function.
            @tf.function
            def cal_loss():
                return cal_loss_dictionary(gains_real=gain_r, gains_imag=gain_i, ants_map=ants_map,
                                           foreground_coeffs_real=fg_r, foreground_coeffs_imag=fg_i,
                                           data_real=data_r, data_imag=data_i, wgts=wgts, foreground_range_map=foreground_range_map,
                                           components_map=model_components_map)
            cal_loss_i = cal_loss().numpy()
            echo(f'{datetime.datetime.now()} Performing Gradient Descent. Initial MSE of {loss_i:.2e}...\n', verbose=verbose)
            # perform optimization loop.
            if freeze_model:
                vars = [gain_r, gain_i]
            else:
                vars = [gain_r, gain_i, fg_r, fg_i]
            if notebook_progressbar:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            min_loss = 9e99
            for step in tqdm(range(maxsteps)):
                with tf.GradientTape() as tape:
                    loss = cal_loss()
                grads = tape.gradient(loss, vars)
                opt.apply_gradients(zip(grads, vars))
                fitting_info_t['loss_history'].append(loss.numpy())
                if record_var_history:
                    fitting_info_t['g_r'].append(gain_r.numpy())
                    fitting_info_t['g_i'].append(gain_i.numpy())
                    if not freeze_model:
                        fitting_info_t['fg_r'].append(fg_r.numpy())
                        fitting_info_t['fg_i'].append(fg_i.numpy())
                if fitting_info_t['loss_history'][-1] < min_loss:
                    # store the g_r, g_i, fg_r, fg_i values that minimize loss
                    # in case of overshoot.
                    min_loss = fitting_info_t['loss_history'][-1]
                    g_r_ml = gain_r.value()
                    g_i_ml = gain_i.value()
                    if not freeze_model:
                        fg_r_ml = fg_r.value()
                        fg_i_ml = fg_i.value()
                    else:
                        fg_r_ml = fg_r
                        fg_i_ml = fg_i

                if step >= 1 and np.abs(fitting_info_t['loss_history'][-1] - fitting_info_t['loss_history'][-2]) < tol:
                    break
            # insert model values.
            echo(f'{datetime.datetime.now()} Finished Gradient Descent. MSE of {min_loss:.2e}...\n', verbose=verbose)
            insert_model_into_uvdata_dictionary(uvdata=model, time_index=time_index, polarization=pol, model_components_map=model_components_map,
                                                foreground_coeffs_real=fg_r_ml, foreground_coeffs_imag=fg_i_ml, scale_factor=rmsdata, foreground_range_map=foreground_range_map)
            insert_gains_into_uvcal_dictionary(uvcal=gains, time_index=time_index, polarization=pol, gains_real=gain_r, gains_imag=gain_i, ants_map=ants_map)

            fitting_info_p[time_index] = fitting_info_t
        fitting_info[polnum] = fitting_info_p
        # compute and multiply out scale-factor accounting for overall amplitude and phase degeneracy.
        scale_factor_phase = np.angle(np.mean(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]]))
        scale_factor_abs = np.sqrt(np.mean(np.abs(sky_model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]] / model.data_array[:, :, :, polnum][~uvdata.flag_array[:, :, :, polnum]]) ** 2.))
        scale_factor = scale_factor_abs * np.exp(1j * scale_factor_phase)
        model.data_array[:, :, :, polnum] *= scale_factor
        gains.gain_array[:, :, :, :, polnum] *= (scale_factor) ** -.5

    model_with_gains = utils.apply_gains(model, gains, inverse=True)
    resid.data_array -= model_with_gains.data_array
    resid = utils.apply_gains(resid, gains)
    filtered = copy.deepcopy(model)
    filtered.data_array += resid.data_array

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
    model, resid, filtered, gains, fitted_info = calibrate_and_model_per_baseline_dictionary_method(uvdata=uvdata, foreground_modeling_components=dpss_model_components,
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
