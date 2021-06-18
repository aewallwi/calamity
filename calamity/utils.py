from pyuvdata import UVCal
import numpy as np
import copy
from uvtools import dspec
import datetime
import tensorflow as tf
from . import simple_cov
from .echo import echo


def blank_uvcal_from_uvdata(uvdata):
    """initialize UVCal object with same times, antennas, and frequencies as uvdata.

    Parameters
    ----------
    uvdata: UVData object
        UVData object that you wish to generate a blanck UVCal object from.

    Returns
    -------
    uvcal: UVCal object
        UVCal object with all flags set to False
        and all gains set to unity with same antennas, freqs, jones, and times
        as uvdata.
    """
    uvcal = UVCal()
    uvcal.Nfreqs = uvdata.Nfreqs
    uvcal.Njones = uvdata.Npols
    uvcal.Ntimes = uvdata.Ntimes
    uvcal.Nspws = uvdata.Nspws
    uvcal.history = ""
    uvcal.Nspws = uvdata.Nspws
    uvcal.telescope_name = uvdata.telescope_name
    uvcal.telescope_location = uvdata.telescope_location
    uvcal.Nants_data = uvdata.Nants_data
    uvcal.Nants_telescope = uvdata.Nants_telescope
    uvcal.ant_array = np.asarray(list(set(uvdata.ant_1_array).union(set(uvdata.ant_2_array))))
    uvcal.antenna_names = uvdata.antenna_names
    uvcal.antenna_numbers = uvdata.antenna_numbers
    uvcal.antenna_positions = uvdata.antenna_positions
    uvcal.spw_array = uvdata.spw_array
    uvcal.freq_array = uvdata.freq_array
    uvcal.jones_array = uvdata.polarization_array
    uvcal.time_array = np.unique(uvdata.time_array)
    uvcal.integration_time = np.mean(uvdata.integration_time)
    uvcal.lst_array = np.unique(uvdata.lst_array)
    uvcal.gain_convention = "divide"  # always use divide for this package.
    uvcal.flag_array = np.zeros(
        (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones),
        dtype=np.bool,
    )
    uvcal.quality_array = np.zeros_like(uvcal.flag_array, dtype=np.float64)
    uvcal.x_orientation = uvdata.x_orientation
    uvcal.gain_array = np.ones_like(uvcal.flag_array, dtype=np.complex128)
    uvcal.cal_style = "redundant"
    uvcal.cal_type = "gain"
    return uvcal


def get_redundant_groups_conjugated(uvdata, remove_redundancy=False, tol=1.0, include_autos=False):
    """Get lists of antenna pairs and redundancies in a uvdata set.

    Provides list of antenna pairs and ant-pairs organized in redundant groups
    with the proper ordering of the antenna numbers so that there are no
    non-redundancies by conjugation only.

    Parameters
    ----------
    uvdata: UVData object.
        uvdata to get antpairs and redundant groups from.
    remove_redundancy: bool, optional
        if True, split all baselines into their own redundant groups, effectively
        removeing redundancy but allowing us to handle redundant and non-redundant
        modeling under the same framework.
    tol: float, optional
        baselines are considered redundant if their bl_x, bl_y coords are
        within tol of eachother (units of meters)
        default is 1.0m
    include_autos: bool, optional
        if True, include autocorrelations
        default is False.

    Returns
    -------
    antpairs: list of 2-tuples
        list of ant-pairs with tuples ordered to remove conjugation effects.
    red_grps: list of lists of 2-tuples.
        list of list where each list contains antpair tuples that are ordered
        so that there are no conjugates.
    red_grp_map: dict
        dictionary with ant-pairs in antpairs as keys mapping to the index of the redundant group
        that they are a memeber of.
    lengths: list
        list of float lengths of redundant baselines in meters.

    """
    antpairs = []
    # set up maps between antenna pairs and redundant groups.
    red_grps, _, lengths, conjugates = uvdata.get_redundancies(
        include_conjugates=True, include_autos=include_autos, tol=tol
    )
    # convert to ant pairs
    red_grps = [[uvdata.baseline_to_antnums(bl) for bl in red_grp] for red_grp in red_grps]
    conjugates = [uvdata.baseline_to_antnums(bl) for bl in conjugates]

    ap_data = set(uvdata.get_antpairs())
    # make sure all red_grps in data and all conjugates in data
    red_grps = [[ap for ap in red_grp if ap in ap_data or ap[::-1] in ap_data] for red_grp in red_grps]
    red_grps = [red_grp for red_grp in red_grps if len(red_grp) > 0]
    conjugates = [ap for ap in conjugates if ap in ap_data or ap[::-1] in ap_data]
    # modeify red_grp lists to have conjugated antpairs ordered consistently.
    red_grps_t = []
    for red_grp in red_grps:
        red_grps_t.append([])
        for ap in red_grp:
            if ap in conjugates:
                red_grps_t[-1].append(ap[::-1])
                antpairs.append(ap[::-1])
            else:
                red_grps_t[-1].append(ap)
                antpairs.append(ap)

    red_grps = red_grps_t
    del red_grps_t

    antpairs = set(antpairs)

    # convert all redundancies to redunant groups with length one if
    # remove_redundancy is True.
    if remove_redundancy:
        red_grps_t = []
        for red_grp in red_grps:
            for ap in red_grp:
                red_grps_t.append([ap])
        red_grps = red_grps_t
        del red_grps_t

    red_grp_map = {}
    for ap in antpairs:
        red_grp_map[ap] = np.where([ap in red_grp for red_grp in red_grps])[0][0]

    return antpairs, red_grps, red_grp_map, lengths


def compute_bllens(uvdata, fit_group):
    lengths = []
    for ap in fit_group:
        dinds = uvdata.antpair2ind(ap)
        if len(dinds) == 0:
            dinds = uvdata.antpair2ind(ap[::-1])
        bllen = np.max(np.linalg.norm(uvdata.uvw_array[dinds], axis=1) ** 2.0, axis=0)
        lengths.append(bllen)
    return lengths


def yield_dpss_model_comps_bl_grp(
    antpairs, length, freqs, cache, horizon=1.0, min_dly=0.0, offset=0.0, operator_cache=None
):
    if operator_cache is None:
        operator_cache = {}
    dly = np.ceil(max(min_dly, length / 0.3 * horizon + offset)) / 1e9
    dpss_model_comps = dspec.dpss_operator(
        freqs,
        filter_centers=[0.0],
        filter_half_widths=[dly],
        eigenval_cutoff=[1e-12],
        cache=operator_cache,
    )[0].real
    return dpss_model_comps


def yield_mixed_comps(uvdata, fitting_groups, eigenval_cutoff=1e-10, ant_dly=0.0, verbose=False, dtype=np.float32):
    """Generate modeling components that include jointly modeled baselines.

    Parameters
    ----------
    uvdata: UVData object
        data to model
    fitting_groups: list of tuple of tuples of 2-tuples
        each tuple in list is a list of redundant groups of baselines that we will fit jointly
        with components that span each group.
        groups with a single redundant baseline group will be modeled with dpss vectors.
        Redundancies must have conjugation already taken into account.
    eigenval_cutoff: float, optional
        threshold of eigenvectors to include in modeling components.
    ant_dly: float, optional
        intrinsic chromaticity of antenna elements.
    verbose: bool, optional
        produce text outputs.
    dtype: numpy.dtype
        data type in which to compute model eigenvectors.
    """
    operator_cache = {}
    modeling_vectors = {}
    for fit_group in fitting_groups:
        # yield dpss
        if len(fit_group) == 1:
            bllens = compute_bllens(uvdata, fit_group)
            modeling_vectors[fit_group] = yield_dpss_model_comps_bl_grp(
                freqs=uvdata.freq_array[0],
                antpairs=fit_group[0],
                length=bllens[0],
                offset=ant_dly,
                cache=operator_cache,
                multibl_key=True,
            )

        else:
            modeling_vectors[fit_group] = simple_cov.yield_multi_baseline_model_comps(
                uvdata=uvdata, antpairs=fit_group, ant_dly=ant_dly, dtype=dtype
            )
    return modeling_vectors


def yield_pbl_dpss_model_comps(
    uvdata,
    horizon=1.0,
    offset=0.0,
    min_dly=0.0,
    include_autos=False,
    verbose=False,
    red_tol=1.0,
):
    """Get dictionary of DPSS vectors for modeling 21cm foregrounds.

    Parameters
    ----------
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
        tolarance for treating baselines as redundant.
        default is 1.0

    Returns
    -------
    dpss_model_comps: dict
        dictionary with redundant group antenna-pair keys pointing to dpss operators.

    """
    dpss_model_comps = {}
    operator_cache = {}
    # generate dpss modeling vectors.
    antpairs, red_grps, red_grp_map, lengths = get_redundant_groups_conjugated(
        uvdata, include_autos=include_autos, tol=red_tol
    )
    echo(
        f"{datetime.datetime.now()} Building DPSS modeling vectors...\n",
        verbose=verbose,
    )

    for red_grp, length in zip(red_grps, lengths):
        dpss_model_comps[(tuple(red_grp),)] = yield_dpss_model_comps_bl_grp(
            antpairs=red_grp,
            length=length,
            freqs=uvdata.freq_array[0],
            cache=operator_cache,
            horizon=1.0,
            min_dly=min_dly,
            offset=offset,
        )

    return dpss_model_comps


def apply_gains(uvdata, gains, inverse=False):
    """apply gains to a uvdata object.

    Parameters
    ----------
    uvdata: UVData object.
        UVData for data to have gains applied.
    gains: UVCal object.
        UVCal object containing gains to be applied.
    inverse: bool, optional
        Multiply gains instead of dividing.
    Returns
    -------
    calibrated: UVData object.
        UVData object containing calibrated data.
    """
    calibrated = copy.deepcopy(uvdata)
    for ap in calibrated.get_antpairs():
        for pnum, pol in enumerate(uvdata.get_pols()):
            dinds = calibrated.antpair2ind(ap)
            if not inverse:
                calibrated.data_array[dinds, 0, :, pnum] = (
                    calibrated.data_array[dinds, 0, :, pnum]
                    / (gains.get_gains(ap[0], "J" + pol) * np.conj(gains.get_gains(ap[1], "J" + pol))).T
                )
            else:
                calibrated.data_array[dinds, 0, :, pnum] = (
                    calibrated.data_array[dinds, 0, :, pnum]
                    * (gains.get_gains(ap[0], "J" + pol) * np.conj(gains.get_gains(ap[1], "J" + pol))).T
                )
            calibrated.flag_array[dinds, 0, :, pnum] = (
                calibrated.flag_array[dinds, 0, :, pnum]
                | (gains.get_flags(ap[0], "J" + pol) | gains.get_flags(ap[1], "J" + pol)).T
            )
    return calibrated
