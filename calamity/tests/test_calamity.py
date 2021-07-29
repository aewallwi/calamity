from ..data import DATA_PATH
import pytest
from .. import calamity
from .. import utils
from .. import modeling
from .. import cal_utils
from pyuvdata import UVData
from pyuvdata import UVCal
import os
import numpy as np
import copy
import tensorflow as tf
from pyuvdata import UVFlag
import sys
import glob


@pytest.fixture
def sky_model():
    uvd = UVData()
    uvd.read_uvh5(
        os.path.join(
            DATA_PATH,
            "Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_gsm.uvh5",
        )
    )
    uvd.select(bls=[ap for ap in uvd.get_antpairs() if ap[0] != ap[1]], inplace=True)
    return uvd


@pytest.fixture
def sky_model_redundant():
    uvd = UVData()
    uvd.read_uvh5(os.path.join(DATA_PATH, "garray_3ant_2_copies_ntimes_1compressed_False_autosTrue_eor_0.0dB.uvh5"))
    uvd.select(bls=[ap for ap in uvd.get_antpairs() if ap[0] != ap[1]], inplace=True)
    return uvd


@pytest.fixture
def gains(sky_model):
    return cal_utils.blank_uvcal_from_uvdata(sky_model)


@pytest.fixture
def gains_redundant(sky_model_redundant):
    return cal_utils.blank_uvcal_from_uvdata(sky_model_redundant)


@pytest.fixture
def weights(sky_model):
    uvf = UVFlag(sky_model, mode="flag")
    uvf.weights_array = np.ones_like(uvf.flag_array).astype(np.float)
    return uvf


@pytest.fixture
def weights_redundant(sky_model_redundant):
    uvf = UVFlag(sky_model_redundant, mode="flag")
    uvf.weights_array = np.ones_like(uvf.flag_array).astype(np.float)
    return uvf


@pytest.fixture
def gains_randomized(gains):
    gains.gain_array += 1e-2 * np.random.randn(*gains.gain_array.shape) + 1e-2j * np.random.randn(
        *gains.gain_array.shape
    )
    return gains


@pytest.fixture
def gains_randomized_redundant(gains_redundant):
    gains_redundant.gain_array += 1e-2 * np.random.randn(*gains_redundant.gain_array.shape) + 1e-2j * np.random.randn(
        *gains_redundant.gain_array.shape
    )
    return gains_redundant


@pytest.fixture
def gains_antscale(gains):
    for i, antnum in enumerate(gains.ant_array):
        gains.gain_array[i] *= antnum + 1.0
    return gains


@pytest.fixture
def gains_antscale_randomized(gains_randomized):
    for i, antnum in enumerate(gains_randomized.ant_array):
        gains_randomized.gain_array[i] *= antnum + 1.0
    return gains_randomized


@pytest.fixture
def dpss_vectors(sky_model):
    return modeling.yield_pbl_dpss_model_comps(sky_model, offset=2.0 / 0.3, min_dly=2.0 / 0.3)


@pytest.fixture
def mixed_vectors(sky_model):
    fitting_grps, blvecs, _, _ = modeling.get_uv_overlapping_grps_conjugated(sky_model)
    return modeling.yield_mixed_comps(
        fitting_grps, blvecs, sky_model.freq_array[0], ant_dly=2.0 / 0.3, grp_size_threshold=1
    )


@pytest.fixture
def mixed_vectors_redundant(sky_model_redundant):
    fitting_grps, blvecs, _, _ = modeling.get_uv_overlapping_grps_conjugated(sky_model_redundant)
    return modeling.yield_mixed_comps(
        fitting_grps, blvecs, sky_model_redundant.freq_array[0], ant_dly=2.0 / 0.3, grp_size_threshold=1
    )


@pytest.fixture
def dpss_vectors_redundant(sky_model_redundant):
    return modeling.yield_pbl_dpss_model_comps(sky_model_redundant, offset=2.0 / 0.3, min_dly=2.0 / 0.3)


@pytest.fixture
def sky_model_projected(sky_model, dpss_vectors):
    for ap in sky_model.get_antpairs():
        dinds = sky_model.antpair2ind(ap)
        if ((ap,),) not in dpss_vectors:
            ap = ap[::-1]
        apk = ((ap,),)
        sky_model.data_array[dinds, 0, :, 0] = (
            dpss_vectors[apk] @ (sky_model.data_array[dinds, 0, :, 0] @ dpss_vectors[apk]).T
        ).T
    return sky_model


@pytest.fixture
def sky_model_projected_redundant(sky_model_redundant, dpss_vectors_redundant):
    for ap in sky_model_redundant.get_antpairs():
        dinds = sky_model_redundant.antpair2ind(ap)
        if ((ap,),) not in dpss_vectors_redundant:
            ap = ap[::-1]
        apk = ((ap,),)
        sky_model_redundant.data_array[dinds, 0, :, 0] = (
            dpss_vectors_redundant[apk]
            @ (sky_model_redundant.data_array[dinds, 0, :, 0] @ dpss_vectors_redundant[apk]).T
        ).T
    return sky_model_redundant


@pytest.fixture
def redundant_groups(sky_model):
    return modeling.get_redundant_grps_conjugated(sky_model)[1]


@pytest.fixture
def uvdata(sky_model_projected):
    uvd = UVData()
    uvd.read_uvh5(
        os.path.join(
            DATA_PATH,
            "Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_eor_-50.0dB.uvh5",
        )
    )
    uvd.data_array = uvd.data_array + sky_model_projected.data_array
    return uvd


@pytest.fixture
def uvdata_redundant(sky_model_projected_redundant):
    uvd = UVData()
    uvd.read_uvh5(
        os.path.join(
            DATA_PATH,
            "garray_3ant_2_copies_ntimes_1compressed_False_autosTrue_eor_0.0dB.uvh5",
        )
    )
    uvd.select(bls=[ap for ap in uvd.get_antpairs() if ap[0] != ap[1]])
    uvd.data_array *= (
        1e-4
        / np.sqrt(np.mean(np.abs(uvd.data_array) ** 2.0))
        * np.sqrt(np.mean(np.abs(sky_model_projected_redundant.data_array) ** 2.0))
    )
    uvd.data_array = uvd.data_array + sky_model_projected_redundant.data_array
    return uvd


def test_renormalize(sky_model, gains):
    gains.gain_array *= (51.0 + 23j) ** -0.5
    sky_model_ref = copy.deepcopy(sky_model)
    sky_model.data_array *= 51.0 + 23j
    assert not np.allclose(gains.gain_array, 1.0)
    assert not np.allclose(sky_model_ref.data_array, sky_model.data_array)
    calamity.renormalize(sky_model_ref, sky_model, gains, polarization="xx")
    assert np.allclose(gains.gain_array, 1.0)
    assert np.allclose(sky_model_ref.data_array, sky_model.data_array)


def test_tensorize_gains(gains_antscale):
    gains_r, gains_i = calamity.tensorize_gains(gains_antscale, polarization="xx", time_index=0, dtype=np.float64)
    assert gains_r.dtype == np.float64
    assert gains_i.dtype == np.float64
    for i, ant in enumerate(gains_antscale.ant_array):
        assert np.allclose(gains_r.numpy()[ant], ant + 1)
        assert np.allclose(gains_i.numpy()[ant], 0.0)


def test_tensorize_fg_model_comps_dpsss(
    sky_model_projected,
    dpss_vectors,
    redundant_groups,
    gains,
):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor, corr_inds = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=sky_model_projected.Nfreqs,
    )
    nfreqs = sky_model_projected.Nfreqs
    bls = list(dpss_vectors.keys())
    ncomps = 0
    for cnum in range(len(corr_inds)):
        for gnum in range(len(corr_inds[cnum])):
            for blnum, bl in enumerate(corr_inds[cnum][gnum]):
                fg_comps_tensor = fg_comp_tensor[cnum][:, gnum, blnum].numpy().squeeze()
                if ((bl,),) in dpss_vectors:
                    fg_comps_dict = dpss_vectors[((bl,),)].T
                    assert np.allclose(fg_comps_dict, fg_comps_tensor[: fg_comps_dict.shape[0]])
                    assert np.allclose(0.0, fg_comps_tensor[fg_comps_dict.shape[0] :])
                    ncomps += 1
                else:
                    assert np.allclose(fg_comps_tensor, 0.0)
    assert ncomps == len(dpss_vectors)


def test_chunk_fg_comp_dict_by_nbls(dpss_vectors):
    dpss_vectors_chunked = calamity.chunk_fg_comp_dict_by_nbls(dpss_vectors)
    maxvecs = np.max([dpss_vectors[k].shape[1] for k in dpss_vectors])
    assert len(dpss_vectors_chunked) == 1
    assert list(dpss_vectors_chunked.keys())[0] == (1, maxvecs)


@pytest.mark.parametrize(
    "redundant_data",
    [True, False],
)
def test_tensorize_fg_model_comps_mixed(
    gains,
    gains_redundant,
    sky_model_projected_redundant,
    mixed_vectors,
    mixed_vectors_redundant,
    redundant_data,
    sky_model,
    sky_model_projected,
):

    if redundant_data:
        gains = gains_redundant
        sky_model = sky_model_projected_redundant
        fg_comps_dict = mixed_vectors_redundant
    else:
        gains = gains
        sky_model = sky_model_projected
        fg_comps_dict = mixed_vectors

    nfreqs = sky_model.Nfreqs
    nants = sky_model.Nants_data

    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensors, corr_inds = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=fg_comps_dict,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=nfreqs,
    )

    cnum = 0
    ndata_chunked = np.sum([fgc.shape[3] * fgc.shape[2] * fgc.shape[1] for fgc in fg_comp_tensors])
    assert ndata_chunked == sky_model.Nbls * sky_model.Nfreqs
    # hash all fitgrps to i, j
    fitgrps = {}
    redgrps = {}
    for fitgrp in fg_comps_dict:
        for redgrp in fitgrp:
            for ap in redgrp:
                fitgrps[ap] = fitgrp
                redgrps[ap] = redgrp
    for cnum in range(len(corr_inds)):
        for gnum in range(len(corr_inds[cnum])):
            for blnum, bl in enumerate(corr_inds[cnum][gnum]):
                fg_comps_tensor = fg_comp_tensors[cnum][:, gnum, blnum].numpy().squeeze()
                if bl in fitgrps:
                    fg_comps_dict_mat = fg_comps_dict[fitgrps[bl]].T
                    redgrpnum = fitgrps[bl].index(redgrps[bl])
                    dslice = slice(redgrpnum * nfreqs, (redgrpnum + 1) * nfreqs)
                    assert np.allclose(fg_comps_dict_mat[:, dslice], fg_comps_tensor[: fg_comps_dict_mat.shape[0]])
                    assert np.allclose(0.0, fg_comps_tensor[fg_comps_dict_mat.shape[0] :])
                else:
                    assert np.allclose(fg_comps_tensor, 0.0)


@pytest.mark.parametrize(
    "redundant_modeling, redundant_data",
    [
        (False, False),
        (False, True),
        (True, True),
        (True, False),
    ],
)
def test_yield_fg_model_and_fg_coeffs_mixed(
    mixed_vectors_redundant,
    mixed_vectors,
    sky_model_projected_redundant,
    sky_model_projected,
    gains_redundant,
    gains,
    redundant_modeling,
    redundant_data,
):
    # loop test where we create a sparse matrix representation of our foregrounds using dpss vector projection
    # and then translate model back into visibility spectra and compare with original visibilties.
    # First, generate sparse matrix representation (sparse foreground components and coefficients).
    if redundant_data:
        gains = gains_redundant
        sky_model = sky_model_projected_redundant
        fg_comps_dict = mixed_vectors_redundant
    else:
        gains = gains
        sky_model = sky_model_projected
        fg_comps_dict = mixed_vectors

    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    nfreqs = sky_model.Nfreqs
    nants = sky_model.Nants_data

    fg_comp_tensors_chunked, corr_inds_chunked = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=fg_comps_dict,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=nfreqs,
        use_redundancy=redundant_modeling,
    )
    data_r, data_i, wgts = calamity.tensorize_data(
        sky_model, corr_inds_chunked, ants_map, polarization="xx", time_index=0, dtype=np.float64
    )

    fg_coeffs_chunked_re = calamity.tensorize_fg_coeffs(data_r, wgts, fg_comp_tensors_chunked)
    fg_coeffs_chunked_im = calamity.tensorize_fg_coeffs(data_i, wgts, fg_comp_tensors_chunked)
    # now retrieve Nants x Nants x Nfreq complex visibility cube from representation.
    model_r = calamity.yield_fg_model_array(
        fg_model_comps=fg_comp_tensors_chunked,
        fg_coeffs=fg_coeffs_chunked_re,
        corr_inds=corr_inds_chunked,
        nants=nants,
        nfreqs=nfreqs,
    )
    model_i = calamity.yield_fg_model_array(
        fg_model_comps=fg_comp_tensors_chunked,
        fg_coeffs=fg_coeffs_chunked_im,
        corr_inds=corr_inds_chunked,
        nants=nants,
        nfreqs=nfreqs,
    )
    model = model_r + 1j * model_i
    # and check that the columns in that cube line up with data.
    for fit_grp in fg_comps_dict:
        for red_grp in fit_grp:
            for ap in red_grp:
                i, j = ants_map[ap[0]], ants_map[ap[1]]
                ap_data = sky_model.get_data(ap + ("xx",))
                ap_model = model[i, j]
                rmsdata = np.mean(np.abs(ap_data) ** 2.0) ** 0.5
                assert np.allclose(ap_model, ap_data, rtol=0.0, atol=1e-2 * rmsdata)


def test_insert_model_into_uvdata_tensor(redundant_groups, dpss_vectors, sky_model_projected, gains):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comps_tensor, corr_inds = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=sky_model_projected.Nfreqs,
    )
    rmsdata = np.mean(np.abs(sky_model_projected.data_array) ** 2.0) ** 0.5
    data_r, data_i, wgts = calamity.tensorize_data(
        sky_model_projected,
        corr_inds,
        ants_map,
        polarization="xx",
        time_index=0,
        dtype=np.float64,
        data_scale_factor=rmsdata,
    )

    fg_coeffs_re = calamity.tensorize_fg_coeffs(data_r, wgts, fg_comps_tensor)
    fg_coeffs_im = calamity.tensorize_fg_coeffs(data_i, wgts, fg_comps_tensor)
    inserted_model = copy.deepcopy(sky_model_projected)
    # set data array to be noise.
    inserted_model.data_array = np.random.randn(*inserted_model.data_array.shape) + 1j * np.random.randn(
        *inserted_model.data_array.shape
    )
    # get model tensor.
    nants = sky_model_projected.Nants_data
    nfreqs = sky_model_projected.Nfreqs
    model_r = calamity.yield_fg_model_array(
        fg_model_comps=fg_comps_tensor, fg_coeffs=fg_coeffs_re, nants=nants, nfreqs=nfreqs, corr_inds=corr_inds
    )
    model_i = calamity.yield_fg_model_array(
        fg_model_comps=fg_comps_tensor, fg_coeffs=fg_coeffs_im, nants=nants, nfreqs=nfreqs, corr_inds=corr_inds
    )
    # insert tensors
    calamity.insert_model_into_uvdata_tensor(
        inserted_model,
        0,
        "xx",
        ants_map,
        redundant_groups,
        model_r,
        model_i,
        scale_factor=rmsdata,
    )
    # check that data arrays are equal
    assert np.allclose(inserted_model.data_array, sky_model_projected.data_array)


@pytest.mark.parametrize(
    "noweights, perfect_data, use_min",
    [
        (True, True, False),
        (True, False, False),
        (False, False, True),
        (True, True, False),
    ],
)
def test_calibrate_and_model_dpss(
    uvdata, sky_model_projected, gains_randomized, gains, weights, noweights, perfect_data, use_min
):
    if noweights:
        weight = None
    else:
        weight = weights
    # check that resid is much smaller then model and original data.
    if perfect_data:
        model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=sky_model_projected,
            gains=gains,
            verbose=True,
            use_redundancy=False,
            sky_model=None,
            maxsteps=3000,
            tol=1e-10,
            correct_resid=True,
            correct_model=True,
            weights=weight,
            use_min=use_min,
        )
    else:
        model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=uvdata,
            gains=gains_randomized,
            verbose=True,
            use_redundancy=False,
            sky_model=None,
            maxsteps=3000,
            tol=1e-10,
            correct_resid=True,
            correct_model=True,
            weights=weight,
            use_min=use_min,
        )
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


@pytest.mark.parametrize(
    "use_redundancy, graph_mode",
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_calibrate_and_model_dpss_redundant(
    uvdata_redundant,
    sky_model_projected_redundant,
    gains_randomized_redundant,
    weights_redundant,
    use_redundancy,
    graph_mode,
):
    model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
        min_dly=2.0 / 0.3,
        offset=2.0 / 0.3,
        uvdata=uvdata_redundant,
        gains=gains_randomized_redundant,
        verbose=True,
        use_redundancy=use_redundancy,
        sky_model=None,
        maxsteps=3000,
        tol=1e-10,
        correct_resid=False,
        correct_model=False,
        graph_mode=graph_mode,
    )

    # post hoc correction
    resid = cal_utils.apply_gains(resid, gains)
    model = cal_utils.apply_gains(model, gains)
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata_redundant.data_array) ** 2.0)) >= 1e2 * np.sqrt(
        np.mean(np.abs(resid.data_array) ** 2.0)
    )
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


def test_calibrate_and_model_dpss_dont_correct_resid(
    uvdata,
    sky_model_projected,
    gains_randomized,
    weights,
):
    # check that resid is much smaller then model and original data.
    model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
        min_dly=2.0 / 0.3,
        offset=2.0 / 0.3,
        uvdata=uvdata,
        gains=gains_randomized,
        verbose=True,
        use_redundancy=False,
        sky_model=None,
        maxsteps=3000,
        tol=1e-10,
        correct_resid=False,
        correct_model=False,
        weights=weights,
    )

    # post hoc correction
    resid = cal_utils.apply_gains(resid, gains)
    model = cal_utils.apply_gains(model, gains)
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


def test_calibrate_and_model_dpss_freeze_model(uvdata, sky_model_projected, gains_randomized, weights):
    # test that calibrating with a perfect sky model and only optimizing gains yields nearly perfect solutions for the gains.
    model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
        min_dly=2.0 / 0.3,
        offset=2.0 / 0.3,
        uvdata=sky_model_projected,
        gains=gains_randomized,
        verbose=True,
        use_redundancy=False,
        sky_model=sky_model_projected,
        freeze_model=True,
        maxsteps=3000,
        tol=1e-10,
        correct_resid=True,
        correct_model=True,
        weights=weights,
    )
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.allclose(
        model.data_array,
        sky_model_projected.data_array,
        atol=1e-5 * np.mean(np.abs(model.data_array) ** 2.0) ** 0.5,
    )
    assert np.allclose(gains.gain_array, gains_randomized.gain_array, rtol=0.0, atol=1e-4)
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


@pytest.mark.parametrize(
    "use_tensorflow, n_profile_steps",
    [(True, 10), (False, 0)],
)
def test_calibrate_and_model_mixed(
    tmpdir, uvdata, sky_model_projected, gains_randomized, weights, use_tensorflow, n_profile_steps
):
    tmp_path = tmpdir.strpath
    logdir = os.path.join(tmp_path, "logdir")
    # check that mixec components and dpss components give similar resids
    model, resid, gains, fit_history = calamity.calibrate_and_model_mixed(
        min_dly=0.0,
        offset=0.0,
        ant_dly=2.0 / 3.0,
        red_tol_freq=0.5,
        uvdata=uvdata,
        gains=gains_randomized,
        verbose=True,
        use_redundancy=False,
        sky_model=None,
        freeze_model=True,
        maxsteps=3000,
        tol=1e-10,
        correct_resid=False,
        correct_model=False,
        weights=weights,
        use_tensorflow_to_derive_modeling_comps=use_tensorflow,
        grp_size_threshold=1,
        graph_mode=True,
        n_profile_steps=n_profile_steps,
        profile_log_dir=logdir,
    )
    # post hoc correction
    resid = cal_utils.apply_gains(resid, gains)
    model = cal_utils.apply_gains(model, gains)
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1
    # check for profiler outputs.
    if n_profile_steps > 0:
        assert os.path.exists(logdir)
        assert len(glob.glob(logdir + "/*")) > 0


@pytest.mark.parametrize(
    "perfect_data",
    [True, False],
)
def test_calibrate_and_model_mixed_redundant(
    uvdata_redundant,
    sky_model_projected_redundant,
    gains_randomized_redundant,
    weights_redundant,
    perfect_data,
    gains_redundant,
):
    if not perfect_data:
        # check that mixec components and dpss components give similar resids
        model, resid, gains, fit_history = calamity.calibrate_and_model_mixed(
            min_dly=0.0,
            offset=0.0,
            ant_dly=2.0 / 0.3,
            red_tol_freq=0.5,
            uvdata=uvdata_redundant,
            gains=gains_randomized_redundant,
            verbose=True,
            use_redundancy=False,
            sky_model=None,
            freeze_model=True,
            maxsteps=3000,
            correct_resid=False,
            correct_model=False,
            weights=weights_redundant,
            use_tensorflow_to_derive_modeling_comps=True,
        )
    else:
        model, resid, gains, fit_history = calamity.calibrate_and_model_mixed(
            min_dly=0.0,
            offset=0.0,
            ant_dly=2.0 / 0.3,
            red_tol_freq=0.5,
            uvdata=sky_model_projected_redundant,
            sky_model=sky_model_projected_redundant,
            gains=None,
            verbose=True,
            use_redundancy=False,
            freeze_model=True,
            maxsteps=3000,
            correct_resid=False,
            correct_model=False,
            weights=weights_redundant,
            use_tensorflow_to_derive_modeling_comps=True,
        )
    # post hoc correction
    # resid = cal_utils.apply_gains(resid, gains)
    # model = cal_utils.apply_gains(model, gains)
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata_redundant.data_array) ** 2.0)) >= 1e2 * np.sqrt(
        np.mean(np.abs(resid.data_array) ** 2.0)
    )
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1
