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


@pytest.fixture
def sky_model():
    uvd = UVData()
    uvd.read_uvh5(
        os.path.join(
            DATA_PATH,
            "Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_gsm.uvh5",
        )
    )
    return uvd


@pytest.fixture
def sky_model_redundant():
    uvd = UVData()
    uvd.read_uvh5(os.path.join(DATA_PATH, "garray_3ant_2_copies_ntimes_1compressed_False_autosTrue_eor_0.0dB.uvh5"))
    uvd.select(bls=[ap for ap in uvd.get_antpairs() if ap[0] != ap[1]])
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
    fgroups, blvecs, _, _ = get_uv_overlapping_grps_conjugated()
    return modeling.yield_mixed_comps()


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


@pytest.mark.parametrize("single_bls_as_sparse", [True, False])
def test_tensorize_fg_model_comps_dpsss(
    sky_model_projected, dpss_vectors, redundant_groups, gains, single_bls_as_sparse
):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor, fg_comp_tensor_chunked, corr_inds_chunked = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=sky_model_projected.Nfreqs,
        single_bls_as_sparse=single_bls_as_sparse,
    )
    if single_bls_as_sparse:
        # retrieve dense numpy array from sparse tensor.
        # check whether values agree with original dpss vectors.
        fg_comp_tensor = tf.sparse.to_dense(fg_comp_tensor).numpy()
        # iterate through and select out nonzero elements and check that they are close to dpss_vectors
        start_index = 0
        for vnum, red_grp in enumerate(dpss_vectors.keys()):
            vdpss = dpss_vectors[red_grp]
            end_index = start_index + vdpss.shape[1]
            for i, j in enumerate(range(start_index, end_index)):
                vreduced = fg_comp_tensor[:, j]
                vreduced = vreduced[np.abs(vreduced) > 0]
                assert np.allclose(vreduced, vdpss[:, i])
            start_index = end_index
    else:
        nfreqs = sky_model_projected.Nfreqs
        bls = list(dpss_vectors.keys())
        for gnum, red_grp in enumerate(dpss_vectors.keys()):
            bl = red_grp[0][0]
            data_inds = corr_inds_chunked[gnum]
            blind = bl[0] * sky_model_projected.Nants_data + bl[1]
            assert np.allclose(data_inds, [bl])
            assert np.allclose(fg_comp_tensor_chunked[gnum].numpy(), dpss_vectors[red_grp])


@pytest.mark.parametrize("single_bls_as_sparse", [True, False])
def test_tensorize_fg_model_comps_mixed(
    sky_model_projected, dpss_vectors, redundant_groups, gains, single_bls_as_sparse
):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor, fg_comp_tensor_chunked, corr_inds_chunked = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=sky_model_projected.Nfreqs,
        single_bls_as_sparse=single_bls_as_sparse,
    )
    if single_bls_as_sparse:
        # retrieve dense numpy array from sparse tensor.
        # check whether values agree with original dpss vectors.
        fg_comp_tensor = tf.sparse.to_dense(fg_comp_tensor).numpy()
        # iterate through and select out nonzero elements and check that they are close to dpss_vectors
        start_index = 0
        for vnum, red_grp in enumerate(dpss_vectors.keys()):
            vdpss = dpss_vectors[red_grp]
            end_index = start_index + vdpss.shape[1]
            for i, j in enumerate(range(start_index, end_index)):
                vreduced = fg_comp_tensor[:, j]
                vreduced = vreduced[np.abs(vreduced) > 0]
                assert np.allclose(vreduced, vdpss[:, i])
            start_index = end_index
    else:
        nfreqs = sky_model_projected.Nfreqs
        bls = list(dpss_vectors.keys())
        for gnum, red_grp in enumerate(dpss_vectors.keys()):
            bl = red_grp[0][0]
            data_inds = corr_inds_chunked[gnum]
            blind = bl[0] * sky_model_projected.Nants_data + bl[1]
            assert np.allclose(np.asarray(data_inds), np.asarray([[bl[0], bl[1]]]))
            assert np.allclose(fg_comp_tensor_chunked[gnum].numpy(), dpss_vectors[red_grp])


def test_yield_fg_model_and_fg_coeffs_sparse_tensor(dpss_vectors, redundant_groups, sky_model_projected, gains):
    # loop test where we create a sparse matrix representation of our foregrounds using dpss vector projection
    # and then translate model back into visibility spectra and compare with original visibilties.
    # First, generate sparse matrix representation (sparse foreground components and coefficients).
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor, _, _ = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors, ants_map=ants_map, dtype=np.float64, nfreqs=sky_model_projected.Nfreqs
    )
    (fg_coeffs_re, fg_coeffs_im, _, _) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        time_index=0,
        polarization="xx",
        dtype=np.float64,
    )
    # now retrieve Nants x Nants x Nfreq complex visibility cube from representation.
    model_r = calamity.yield_fg_model_array(
        fg_comps_sparse=fg_comp_tensor,
        fg_coeffs_sparse=fg_coeffs_re,
        nants=sky_model_projected.Nants_data,
        nfreqs=sky_model_projected.Nfreqs,
    )
    model_i = calamity.yield_fg_model_array(
        fg_comps_sparse=fg_comp_tensor,
        fg_coeffs_sparse=fg_coeffs_im,
        nants=sky_model_projected.Nants_data,
        nfreqs=sky_model_projected.Nfreqs,
    )
    model = model_r + 1j * model_i
    # and check that the columns in that cube line up with data.
    for grp in redundant_groups:
        for ap in grp:
            i, j = ants_map[ap[0]], ants_map[ap[1]]
            ap_data = sky_model_projected.get_data(ap + ("xx",))
            ap_model = model[i, j]
            rmsdata = np.mean(np.abs(ap_data) ** 2.0) ** 0.5
            assert np.allclose(ap_model, ap_data, rtol=0.0, atol=1e-5 * rmsdata)


def test_insert_model_into_uvdata_tensor(redundant_groups, dpss_vectors, sky_model_projected, gains):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comps_tensor, _, _ = calamity.tensorize_fg_model_comps_dict(
        fg_model_comps_dict=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
        nfreqs=sky_model_projected.Nfreqs,
    )
    (fg_coeffs_re, fg_coeffs_im, _, _) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        time_index=0,
        polarization="xx",
        dtype=np.float64,
    )
    inserted_model = copy.deepcopy(sky_model_projected)
    # set data array to be noise.
    inserted_model.data_array = np.random.randn(*inserted_model.data_array.shape) + 1j * np.random.randn(
        *inserted_model.data_array.shape
    )
    # get model tensor.
    nants = sky_model_projected.Nants_data
    nfreqs = sky_model_projected.Nfreqs
    model_r = calamity.yield_fg_model_array(
        fg_comps_sparse=fg_comps_tensor, fg_coeffs_sparse=fg_coeffs_re, nants=nants, nfreqs=nfreqs
    )
    model_i = calamity.yield_fg_model_array(
        fg_comps_sparse=fg_comps_tensor, fg_coeffs_sparse=fg_coeffs_im, nants=nants, nfreqs=nfreqs
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
    )
    # check that data arrays are equal
    assert np.allclose(inserted_model.data_array, sky_model_projected.data_array)


def test_tensorize_data(sky_model_projected, redundant_groups, gains):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    for tnum in range(sky_model_projected.Ntimes):
        data_re, data_im, wgts = calamity.tensorize_data(
            sky_model_projected, redundant_groups, ants_map, "xx", 0, dtype=np.float64
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                i, j = ants_map[ap[0]], ants_map[ap[1]]
                data = sky_model_projected.get_data(ap + ("xx",))[tnum]
                assert np.allclose(data, data_re[i, j].numpy() + 1j * data_im[i, j].numpy())
                assert np.allclose(
                    wgts[i, j], np.sum(sky_model_projected.nsample_array * ~sky_model_projected.flag_array) ** -1.0
                )


@pytest.mark.parametrize(
    "use_sparse, noweights, perfect_data",
    [
        (False, True, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
    ],
)
def test_calibrate_and_model_dpss(uvdata, sky_model_projected, gains_randomized, gains, use_sparse, weights, noweights, perfect_data):
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
            single_bls_as_sparse=use_sparse,
            correct_resid=True,
            correct_model=True,
            weights=weight,
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
            single_bls_as_sparse=use_sparse,
            correct_resid=True,
            correct_model=True,
            weights=weight,
        )
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


@pytest.mark.parametrize(
    "use_redundancy, use_sparse",
    [(True, False), (True, True), (False, False), (False, True)],
)
def test_calibrate_and_model_dpss_redundant(
    uvdata_redundant,
    sky_model_projected_redundant,
    gains_randomized_redundant,
    weights_redundant,
    single_bls_as_sparse,
    use_redundancy,
    use_sparse,
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
        single_bls_as_sparse=use_sparse,
        correct_resid=False,
        correct_model=False,
        weights=weights,
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


@pytest.mark.parametrize(
    "use_sparse",
    [True, False],
)
def test_calibrate_and_model_dpss_dont_correct_resid(
    uvdata,
    sky_model_projected,
    gains_randomized,
    use_sparse,
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
        single_bls_as_sparse=use_sparse,
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


@pytest.mark.parametrize(
    "use_sparse",
    [True, False],
)
def test_calibrate_and_model_dpss_freeze_model(uvdata, sky_model_projected, gains_randomized, use_sparse, weights):
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
        single_bls_as_sparse=use_sparse,
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
    "use_tensorflow",
    [True, False],
)
def test_calibrate_and_model_mixed(uvdata, sky_model_projected, gains_randomized, weights, use_tensorflow):

    # check that mixec components and dpss components give similar resids
    model, resid, gains, fit_history = calamity.calibrate_and_model_mixed(
        min_dly=0.0,
        offset=0.0,
        ant_dly=2.0 / 3.0,
        red_tol_freq=0.5,
        uvdata=sky_model_projected,
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
    )
    # post hoc correction
    resid = cal_utils.apply_gains(resid, gains)
    model = cal_utils.apply_gains(model, gains)
    assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e2 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.0))
    assert len(fit_history) == 1
    assert len(fit_history[0]) == 1


@pytest.mark.parametrize(
    "single_bls_as_sparse",
    [False, True],
)
def test_calibrate_and_model_mixed_redundant(
    uvdata_redundant, sky_model_projected_redundant, gains_randomized_redundant, weights_redundant, single_bls_as_sparse
):

    # check that mixec components and dpss components give similar resids
    model, resid, gains, fit_history = calamity.calibrate_and_model_mixed(
        min_dly=0.0,
        offset=0.0,
        ant_dly=2.0 / 0.3,
        red_tol_freq=0.5,
        uvdata=sky_model_projected_redundant,
        gains=gains_randomized_redundant,
        verbose=True,
        use_redundancy=False,
        sky_model=None,
        freeze_model=True,
        maxsteps=3000,
        correct_resid=False,
        correct_model=False,
        weights=weights_redundant,
        single_bls_as_sparse=single_bls_as_sparse,
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
