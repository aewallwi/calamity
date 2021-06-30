import numpy as np
from uvtools import dspec
import datetime
import tensorflow as tf
from . import simple_cov
from .utils import echo


def get_redundant_grps_conjugated(uvdata, remove_redundancy=False, tol=1.0, include_autos=False):
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
    vec_bin_centers: list
        list of float 3-arrays with xyz vector bin centers of each
        redundant group.
    lengths: list
        list of float lengths of redundant baselines in meters.

    """
    antpairs = []
    # set up maps between antenna pairs and redundant groups.
    red_grps, vec_bin_centers, lengths, conjugates = uvdata.get_redundancies(
        include_conjugates=True, include_autos=include_autos, tol=tol
    )
    # convert to ant pairs
    red_grps = [[uvdata.baseline_to_antnums(bl) for bl in red_grp] for red_grp in red_grps]
    conjugates = [uvdata.baseline_to_antnums(bl) for bl in conjugates]

    ap_data = set(uvdata.get_antpairs())
    # make sure all red_grps in data and all conjugates in data
    red_grps = [[ap for ap in red_grp if ap in ap_data or ap[::-1] in ap_data] for red_grp in red_grps]
    lengths = [length for length, red_grp in zip(lengths, red_grps) if len(red_grp) > 0]
    vec_bin_centers = [vbc for vbc, red_grp in zip(vec_bin_centers, red_grps) if len(red_grp) > 0]
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

    return antpairs, red_grps, vec_bin_centers, lengths


def get_uv_overlapping_grps_conjugated(
    uvdata,
    remove_redundancy=False,
    red_tol=1.0,
    include_autos=False,
    red_tol_freq=0.5,
    n_angle_bins=200,
    notebook_progressbar=False,
):
    """Derive groups of baselines that overlap in frequency.

    Parameters
    ----------
    uvdata: UVData object
        uvdata containing data to determine overlapping baseline groups
    remove_redundancy: bool, optional
        If True, all baselines are put in their own redundant groups
    red_tol: float, optional
        distance between baselines for them to be considered redundant
        (units of meters)
    include_autos: bool, optional
        if true, include autocorrelations in redundant groups
    freq_tol: float, optional
        maximum distance between baselines in uv plane at some frequency
        to be placed in the same
    notebook_progressbar: bool, optional
        if True, show graphical notebook progress bar that looks good in jupyter.
        default is False.

    Returns
    -------
    fitting_grps: list
        list of tuples of tuples of 2-tuples. Each tuple is a fitting group
        each tuple in each fitting group is a redundant group
    """
    if notebook_progressbar:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    # first get redundant baselines.
    antpairs, red_grps, vec_bin_centers, lengths = get_redundant_grps_conjugated(
        uvdata, include_autos=include_autos, tol=red_tol
    )
    # next, we build fitting fitting_grps by generating a hashmap of connections between baselines
    fmin = uvdata.freq_array.min()
    fmax = uvdata.freq_array.max()
    vbc_hash = {}
    connections = {}
    # bin baselines into angular bins. Only search for connections within each angular bin.
    grp_nums = {i: [] for i in range(n_angle_bins)}
    dangle = np.pi / n_angle_bins
    grp_num = 0
    for red_grp, vbc in zip(red_grps, vec_bin_centers):
        vbc_hash[tuple(red_grp)] = vbc
        if np.abs(vbc[0]) > 0.0:
            bin_index = int(np.min([np.round((np.arctan(vbc[1] / vbc[0]) + np.pi / 2) / dangle), n_angle_bins - 2]))
        else:
            bin_index = n_angle_bins - 1
        grp_nums[bin_index].append(grp_num)
        grp_num += 1

    for binnum in tqdm(range(n_angle_bins)):
        nums = grp_nums[binnum]
        nnums = len(nums)
        for i in range(nnums):
            grp_num0 = nums[i]
            vbc0 = vec_bin_centers[grp_num0]
            red_grp0 = red_grps[grp_num0]
            if tuple(red_grp0) not in connections:
                connections[tuple(red_grp0)] = set({})
                vbc_hash[tuple(red_grp0)] = vbc0
            for j in range(i + 1, nnums):
                grp_num1 = nums[j]
                red_grp1 = red_grps[grp_num1]
                vbc1 = vec_bin_centers[grp_num1]
                uvwmin0 = fmin * np.linalg.norm(vbc0) / 3e8
                uvwmin1 = fmin * np.linalg.norm(vbc1) / 3e8
                uvwmax0 = fmax * np.linalg.norm(vbc0) / 3e8
                uvwmax1 = fmax * np.linalg.norm(vbc1) / 3e8
                if (uvwmin0 > uvwmin1 and uvwmin0 < uvwmax1) or (uvwmin1 > uvwmin0 and uvwmin1 < uvwmax0):
                    u0 = vbc0[0] * uvdata.freq_array[0] / 3e8
                    v0 = vbc0[1] * uvdata.freq_array[0] / 3e8
                    u1 = vbc1[0] * uvdata.freq_array[0] / 3e8
                    v1 = vbc1[1] * uvdata.freq_array[0] / 3e8
                    ug0, ug1 = np.meshgrid(u0, u1)
                    vg0, vg1 = np.meshgrid(v0, v1)
                    if np.any(np.sqrt(np.abs(ug0 - ug1) ** 2.0 + (vg0 - vg1) ** 2.0) <= red_tol_freq):
                        connections[tuple(red_grp0)].add(tuple(red_grp1))
                        if tuple(red_grp1) not in connections:
                            connections[tuple(red_grp1)] = set({})
                            vbc_hash[tuple(red_grp1)] = vbc1
                        connections[tuple(red_grp1)].add(tuple(red_grp0))
                    elif np.any(np.sqrt(np.abs(ug0 + ug1) ** 2.0 + (vg0 + vg1) ** 2.0) <= red_tol_freq):
                        red_grps[grp_num1] = [ap[::-1] for ap in red_grps[grp_num1]]
                        vec_bin_centers[grp_num1] = [-vbc for vbc in vec_bin_centers[grp_num1]]
                        red_grp1 = red_grps[grp_num1]
                        connections[tuple(red_grp0)].add(tuple(red_grps[grp_num1]))
                        if tuple(red_grp1) not in connections:
                            connections[tuple(red_grp1)] = set({})
                            vbc_hash[tuple(red_grp1)] = vbc1
                        connections[tuple(red_grp1)].add(tuple(red_grp0))

    # now from connections, generate fitting groups.
    fitting_grps = {}  # keeps track of the fitting groups
    bl_lengths = {}  # keep track of baseline vectors in fitting groups
    grp_labels = {}  # keeps track of the label of each group that contains a given redundant set
    # sort connection keys by baseline length and angle
    bl_angles = []
    bl_lengths = []
    red_grps_sorted = []
    for red_grp in vbc_hash:
        bl_lengths.append(np.linalg.norm(vbc_hash[red_grp]))
        bl_angles.append(np.arccos(vbc_hash[red_grp][0] / bl_lengths[-1]))
        red_grps_sorted.append(red_grp)
    # sort now
    red_grps_sorted = sorted(
        red_grps_sorted, key=lambda x: (bl_angles[red_grps_sorted.index(x)], bl_lengths[red_grps_sorted.index(x)])
    )  # ['c003', 'd004', 'b002', 'a001', 'e005']

    for red_grp in tqdm(red_grps_sorted):
        # check if red_grp or any of its connections have already been assigned a group.
        no_connections_in_group = not (red_grp in grp_labels)
        for connection in connections[red_grp]:
            if connection in grp_labels:
                no_existing_connection = False
                connector = connection
                break
        if no_connections_in_group:
            fitting_grps[red_grp] = [red_grp]
            grp_labels[red_grp] = red_grp
            # add all connections of this red baseline group (baselines that have some freq redundancy)
            # to the new fitting group and also record red_grp as the label for their fitting group.
            for connection in connections[red_grp]:
                if connection not in grp_labels:
                    fitting_grps[red_grp].append(connection)
                    grp_labels[connection] = red_grp
        else:
            # if red baseline group has already been assigned a fitting group
            # then add all baselines that are frequency redundant to the
            # parent fitting group.
            parent_grp = grp_labels[red_grp]
            for connection in connections[red_grp]:
                if connection not in grp_labels:
                    fitting_grps[parent_grp].append(connection)
                    grp_labels[connection] = parent_grp

    # convert fitting grps to a list of tuples of tuples of int-2-tuples.
    fitting_grps = list(fitting_grps.values())
    # store vector bin centers in list of tuples of float 3-tuples
    fitting_vec_centers = []
    for fit_grp in fitting_grps:
        fitting_vec_centers.append([])
        for red_grp in fit_grp:
            fitting_vec_centers[-1].append(vbc_hash[red_grp])

    return fitting_grps, fitting_vec_centers, connections, grp_labels


def simple_cov_matrix(blvecs, ant_dly, freqs, dtype=np.float32, use_tensorflow=False):
    """Compute simple covariance matrix for subset of baselines in uvdata

    Parameters
    ----------
    blvecs: list of np.ndarrays
      list of length-3 np.ndarrays representing ENH baseline vectors.
    ant_dly: float
        intrinsic chromaticity of each antenna element.
    freqs: array-like
        array of frequencies (Hz) sampled by interferometer.
    dtype: numpy.dtype, optional
        data type to process and store covariance matrix as.
        default is np.float32
    use_tensorflow: bool, optional
        if True, use tensorflow module sto compute covariance matrix.
        Only recommended for machines with GPUs.
        default is False.

    Returns
    -------
    cmat: tf.Tensor object (if use_tensorflow is True) or np.ndarray(use_tensorflow is False)
        (Nbls * Nfreqs) x (Nbls * Nfreqs) tf.Tensor object or np.ndarray with dtype dtype

    """

    nbls = len(blvecs)
    uvws = np.asarray(blvecs, dtype=dtype)
    freqs = np.asarray(freqs, dtype=dtype)
    nfreqs = len(freqs)
    if use_tensorflow:
        uvws = tf.convert_to_tensor(uvws, dtype=dtype)
        freqs = tf.convert_to_tensor(freqs, dtype)

    absdiff = np.zeros((nbls * nfreqs, nbls * nfreqs), dtype=dtype)
    if use_tensorflow:
        tf.convert_to_tensor(absdiff, dtype=dtype)
    for uvw_ind in range(3):
        if use_tensorflow:
            uvw_coord = tf.reshape(tf.experimental.numpy.outer(uvws[:, uvw_ind], freqs / 3e8), (nbls * nfreqs,))
            cg0, cg1 = tf.meshgrid(uvw_coord, uvw_coord, indexing="ij")
            absdiff += np.abs(cg0 - cg1) ** 2.0
        else:
            uvw_coord = np.reshape(np.outer(uvws[:, uvw_ind], freqs / 3e8), (nbls * nfreqs,))
            cg0, cg1 = np.meshgrid(uvw_coord, uvw_coord, indexing="ij")
            absdiff += np.abs(cg0 - cg1) ** 2.0

    absdiff = absdiff ** 0.5
    if use_tensorflow:
        cmat = tf.experimental.numpy.sinc(2 * absdiff)
    else:
        cmat = np.sinc(2 * absdiff)
    del absdiff
    if use_tensorflow:
        fvals = tf.reshape(tf.experimental.numpy.outer(tf.ones(nbls, dtype=dtype), freqs), (nbls * nfreqs,))
        fg0, fg1 = tf.meshgrid(fvals, fvals, indexing="ij")
        cmat = cmat * tf.experimental.numpy.sinc(2 * tf.abs(fg0 - fg1) * ant_dly)
    else:
        fvals = np.reshape(np.outer(tf.ones(nbls, dtype=dtype), freqs), (nbls * nfreqs,))
        fg0, fg1 = np.meshgrid(fvals, fvals, indexing="ij")
        cmat = cmat * np.sinc(2 * np.abs(fg0 - fg1) * ant_dly)

    del fg0, fg1
    return cmat


def yield_simple_multi_baseline_model_comps(
    blvecs, ant_dly, freqs, dtype=np.float32, eigenval_cutoff=1e-10, verbose=False, use_tensorflow=False
):
    """Generate model components for multiple baselines.

    Parameters
    ----------
    blvecs: list of len-3 numpy.ndarrays
      list of baseline vectors ENH
    ant_dly: float,
      intrinsic delay of antenna (ns).
    freqs: array-like
      array of frequencies sampled by interferometer (MHz).
    dtype: numpy.dtype, optional
      data-type to use for eigenvalue decomposition
      default is np.float32
    eigenval_cutoff: float, optional
        threshold of eigenvectors to include in modeling components.
    verbose: bool, optional
        text outputs
        default is False.
    use_tensorflow: bool, optional
        if True, use tensorflow for multi-baseline modeling matrix operations.
        Recommended only for machines with GPUs.

    Returns
    -------
    multi_baseline_model_comps: np.ndarray
        (Nbl * Nfreqs) x Ncomponents numpy.ndarray containing eigenvectors
        with eigenvalues above cutoff threshold.

    """
    cmat = simple_cov_matrix(blvecs, ant_dly, freqs, dtype=dtype, use_tensorflow=use_tensorflow)
    # get eigenvectors
    echo(
        f"{datetime.datetime.now()} Deriving modeling components with eigenvalue decomposition...\n",
        verbose=verbose,
    )
    if use_tensorflow:
        evals, evecs = tf.linalg.eigh(cmat)
        evals = evals.numpy()
        evecs = evecs.numpy()
    else:
        evals, evecs = np.linalg.eigh(cmat)

    selection = evals / evals.max() >= eigenval_cutoff
    evals = evals[selection][::-1]
    evecs = evecs[:, selection][:, ::-1]
    return evecs


def yield_dpss_model_comps_bl_grp(
    length,
    freqs,
    cache,
    horizon=1.0,
    min_dly=0.0,
    offset=0.0,
    operator_cache=None,
    eigenval_cutoff=1e-10,
):
    """Get per-baseline DPSS modeling vectors

    Parameters
    ----------
    length: float
      length of baseline to model (meters)
    freqs: array-like
      np.ndarray of frequencies
    cache: dict
      dictionary with keys pointing to numpy ndarrays
    horizon: float, optional
      fraction of horizon to model
    min_dly: float, optional
      default is 0.0
    offset: float, optional
      default is 0.0
    operator_cache: dict, optional
      dictionary caching operator matrices
      default is None -> no caching
    eigenval_cutoff: float, optional
      default is 1e-10

    Returns
    -------
    dpss_model_comps: np.ndarray
      Nfreqs x Ncomponents array of floats.
    """
    if operator_cache is None:
        operator_cache = {}
    dly = np.ceil(max(min_dly, length / 0.3 * horizon + offset)) / 1e9
    dpss_model_comps = dspec.dpss_operator(
        freqs,
        filter_centers=[0.0],
        filter_half_widths=[dly],
        eigenval_cutoff=[eigenval_cutoff],
        cache=operator_cache,
    )[0].real
    return dpss_model_comps


def yield_mixed_comps(
    fitting_grps,
    fitting_blvecs,
    freqs,
    eigenval_cutoff=1e-10,
    ant_dly=0.0,
    verbose=False,
    dtype=np.float32,
    notebook_progressbar=False,
    use_tensorflow=False,
):
    """Generate modeling components that include jointly modeled baselines.

    Parameters
    ----------
    fitting_grps: list of tuple of tuples of 2-tuples
        each tuple in list is a list of redundant groups of baselines that we will fit jointly
        with components that span each group.
        groups with a single redundant baseline group will be modeled with dpss vectors.
        Redundancies must have conjugation already taken into account.
    fitting_blvecs: list of lists of len-3 np.ndarrays
        each list element represents a group of redundant baselines we will fit jointly
        and is a list of length 3 np.ndarrays containing the ENH coords of each
        baseline vector.
    freqs: np.ndarray
      list of floats
    eigenval_cutoff: float, optional
      threshold of eigenvectors to include in modeling components.
    ant_dly: float, optional
      intrinsic chromaticity of antenna elements.
    verbose: bool, optional
      produce text outputs.
    dtype: numpy.dtype
      data type in which to compute model eigenvectors.

    Returns
    -------
    modeling_vectors: dict
      dictionary with tuples of tuples of int 2-tuples as keys
      (representing redundant baseline groups) and np.ndarrays as values
      each is (Nfreqs * Ngrp_bls) x Ncomponents

    """
    operator_cache = {}
    modeling_vectors = {}
    if notebook_progressbar:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    for grpnum in tqdm(range(len(fitting_grps))):
        # yield dpss
        fit_grp = fitting_grps[grpnum]
        if isinstance(fit_grp, list):
            fit_grp = tuple(fit_grp)
        blvecs = fitting_blvecs[grpnum]
        bllens = np.linalg.norm(blvecs, axis=1)
        if len(fit_grp) == 1:
            modeling_vectors[fit_grp] = yield_dpss_model_comps_bl_grp(
                freqs=freqs,
                antpairs=fit_grp[0],
                length=bllens[0],
                offset=ant_dly,
                cache=operator_cache,
                eigenval_cutoff=eigenval_cutoff,
            )

        else:
            modeling_vectors[fit_grp] = yield_simple_multi_baseline_model_comps(
                blvecs=blvecs,
                ant_dly=ant_dly,
                dtype=dtype,
                freqs=freqs,
                eigenval_cutoff=eigenval_cutoff,
                use_tensorflow=use_tensorflow,
            )
    return modeling_vectors
