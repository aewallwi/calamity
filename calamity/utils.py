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
    uvcal.history = ''
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
    uvcal.gain_convention = 'divide' # always use divide for this package.
    uvcal.flag_array = np.zeros((uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones), dtype=np.bool)
    uvcal.quality_array = np.zeros_like(uvcal.flag_array, dtype=np.float64)
    uvcal.x_orientation = uvdata.x_orientation
    uvcal.gain_array = np.ones_like(uvcal.flag_array, dtype=np.complex128)
    uvcal.cal_style = "redundant"
    uvcal.cal_type = "gain"
    return uvcal


def get_redundant_groups_conjugated(uvdata, remove_redundancy=False, tol=1.0):
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
        red_grps, _, lengths, conjugates = uvdata.get_redundancies(include_conjugates=True, include_autos=False, tol=tol)
        # convert to ant pairs
        red_grps = [[uvdata.baseline_to_antnums(bl) for bl in red_grp] for red_grp in red_grps]
        conjugates = [uvdata.baseline_to_antnums(bl) for bl in conjugates]

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
