from pyuvdata import UVCal
import numpy as np
import copy


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
    uvcal.time_range = (
        uvcal.time_array.min() - uvcal.integration_time / 2.0,
        uvcal.time_array.max() + uvcal.integration_time / 2.0,
    )
    uvcal.channel_width = np.median(np.diff(uvcal.freq_array))

    return uvcal


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
    for pnum, pol in enumerate(uvdata.get_pols()):
        for ap in calibrated.get_antpairs():
            dinds = calibrated.antpair2ind(ap)
            gindp = np.where(gains.jones_array == uvutils.polstr2num(pol, x_orientation=uvcal.x_orientation))[0][0]
            for time in calibrated.time_array[dinds]:
                dindt = np.where(calibrated.time_array[dinds] == time)[0][0]
                gindt = np.where(np.isclose(gains.time_array, time, rtol=0. atol=1e-6))[0][0]
                aind0 = np.where(gains.ant_array == ap[0])[0][0]
                aind1 = np.where(gains.ant_array == ap[1])[0][0]
                if not inverse:
                    calibrated.data_array[dindt, 0, :, pnum] = (
                        calibrated.data_array[dindt, 0, :, pnum]
                        / (gains.gain_array[aind0, 0, :, gindt, gindp] * np.conj(gains.gain_array[aind1, 0, :, gindt, gindp])
                    )
                else:
                    calibrated.data_array[dindt, 0, :, pnum] = (
                        calibrated.data_array[dindt, 0, :, pnum]
                        * (gains.gain_array[aind0, 0, :, gindt, gindp] * np.conj(gains.gain_array[aind1, 0, :, gindt, gindp])
                    )
                calibrated.flag_array[dindt, 0, :, pnum] = (
                    calibrated.flag_array[dindt, 0, :, pnum]
                    | (gains.flag_array[aind0, 0, :, gindt, gindp] | gains.flag_array[aind1, 0, :, gindt, gindp]
                )
    return calibrated
