from ..data import DATA_PATH
import pytest
from .. import calamity
from pyuvdata import UVData
from pyuvdata import UVCal
from .. import utils
import os
import numpy as np
import copy
import tensorflow as tf


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
def gains(sky_model):
    return utils.blank_uvcal_from_uvdata(sky_model)


@pytest.fixture
def gains_randomized(gains):
    gains.gain_array += 1e-2 * np.random.randn(*gains.gain_array.shape) + 1e-2j * np.random.randn(
        *gains.gain_array.shape
    )
    return gains


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
    return utils.yield_dpss_model_comps(sky_model, offset=2.0 / 0.3, min_dly=2.0 / 0.3)


@pytest.fixture
def sky_model_projected(sky_model, dpss_vectors):
    for ap in sky_model.get_antpairs():
        dinds = sky_model.antpair2ind(ap)
        if ap not in dpss_vectors:
            ap = ap[::-1]
        sky_model.data_array[dinds, 0, :, 0] = (
            dpss_vectors[ap] @ (sky_model.data_array[dinds, 0, :, 0] @ dpss_vectors[ap]).T
        ).T
    return sky_model


@pytest.fixture
def redundant_groups(sky_model):
    return utils.get_redundant_groups_conjugated(sky_model)[1]


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


def test_tensorize_pbl_model_comps_dictionary(sky_model_projected, dpss_vectors, redundant_groups):
    (
        fg_range_map,
        comp_tensor_map,
    ) = calamity.tensorize_pbl_model_comps_dictionary(redundant_groups, dpss_vectors, dtype=np.float64)
    for grp in redundant_groups:
        for ap in grp:
            tdata = sky_model_projected.get_data(ap[0], ap[1], "xx").T
            comps = comp_tensor_map[ap].numpy()
            model = comps @ (tdata.T @ comps).T
            # check that dpss compoments accurately describe sky-model data to
            # the 1e-5 level.
            rmsdata = np.mean(np.abs(tdata) ** 2.0) ** 0.5
            assert np.allclose(model, tdata, rtol=0.0, atol=1e-5 * rmsdata)


def test_sparse_tensorize_pbl_fg_model_comps(sky_model_projected, dpss_vectors, redundant_groups, gains):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor = calamity.sparse_tensorize_pbl_fg_model_comps(
        red_grps=redundant_groups,
        fg_model_comps=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
    )
    # retrieve dense numpy array from sparse tensor.
    # check whether values agree with original dpss vectors.
    fg_comp_tensor = tf.sparse.to_dense(fg_comp_tensor).numpy()
    # iterate through and select out nonzero elements and check that they are close to dpss_vectors
    start_index = 0
    for vnum, red_grp in enumerate(redundant_groups):
        vdpss = dpss_vectors[red_grp[0]]
        end_index = start_index + vdpss.shape[1]
        for i, j in enumerate(range(start_index, end_index)):
            vreduced = fg_comp_tensor[:, j]
            vreduced = vreduced[np.abs(vreduced) > 0]
        assert np.allclose(vreduced, vdpss[:, i])
        start_index = end_index


def test_yield_fg_model_and_fg_coeffs_dictionary(dpss_vectors, redundant_groups, sky_model_projected):
    (
        fg_range_map,
        comp_tensor_map,
    ) = calamity.tensorize_pbl_model_comps_dictionary(redundant_groups, dpss_vectors, dtype=np.float64)
    (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        redundant_groups,
        time_index=0,
        polarization="xx",
        dtype=np.float64,
    )
    # test that all model predictions are close to the underlying data they are derived from.
    for ap in dpss_vectors:
        (model_r, model_i,) = calamity.yield_fg_model_pbl_dictionary_method(
            ap[0],
            ap[1],
            fg_coeffs_re,
            fg_coeffs_im,
            fg_range_map,
            comp_tensor_map,
        )
        model = model_r.numpy() + 1j * model_i.numpy()
        data = sky_model_projected.get_data(ap + ("xx",))
        rmsdata = np.mean(np.abs(data) ** 2.0) ** 0.5

        assert np.allclose(model, data, rtol=0.0, atol=1e-5 * rmsdata)


def test_yield_fg_model_and_fg_coeffs_sparse_tensor(dpss_vectors, redundant_groups, sky_model_projected, gains):
    # loop test where we create a sparse matrix representation of our foregrounds using dpss vector projection
    # and then translate model back into visibility spectra and compare with original visibilties.
    # First, generate sparse matrix representation (sparse foreground components and coefficients).
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comp_tensor = calamity.sparse_tensorize_pbl_fg_model_comps(
        red_grps=redundant_groups,
        fg_model_comps=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
    )
    (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        redundant_groups,
        time_index=0,
        polarization="xx",
        force2d=True,
        dtype=np.float64,
    )
    # now retrieve Nants x Nants x Nfreq complex visibility cube from representation.
    model_r = calamity.yield_fg_pbl_model_sparse_tensor(
        fg_comp_tensor,
        fg_coeffs_re,
        sky_model_projected.Nants_data,
        sky_model_projected.Nfreqs,
    )
    model_i = calamity.yield_fg_pbl_model_sparse_tensor(
        fg_comp_tensor,
        fg_coeffs_im,
        sky_model_projected.Nants_data,
        sky_model_projected.Nfreqs,
    )
    model = model_r.numpy() + 1j * model_i.numpy()
    # and check that the columns in that cube line up with data.
    for grp in redundant_groups:
        for ap in grp:
            i, j = ants_map[ap[0]], ants_map[ap[1]]
            ap_data = sky_model_projected.get_data(ap + ("xx",))
            ap_model = model[i, j]
            rmsdata = np.mean(np.abs(ap_data) ** 2.0) ** 0.5
            assert np.allclose(ap_model, ap_data, rtol=0.0, atol=1e-5 * rmsdata)


def test_insert_model_into_uvdata_dictionary(dpss_vectors, redundant_groups, sky_model_projected):
    (fg_range_map, comp_tensor_map) = calamity.tensorize_pbl_model_comps_dictionary(
        redundant_groups, dpss_vectors, dtype=np.float64
    )
    (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        redundant_groups,
        time_index=0,
        polarization="xx",
    )
    inserted_model = copy.deepcopy(sky_model_projected)
    # set data array to be noise.
    inserted_model.data_array = np.random.randn(*inserted_model.data_array.shape) + 1j * np.random.randn(
        *inserted_model.data_array.shape
    )
    # insert tensors
    calamity.insert_model_into_uvdata_dictionary(
        inserted_model,
        0,
        "xx",
        fg_range_map,
        comp_tensor_map,
        fg_coeffs_re,
        fg_coeffs_im,
    )
    # check that data arrays are equal
    assert np.allclose(inserted_model.data_array, sky_model_projected.data_array)


def test_insert_model_into_uvdata_tensor(redundant_groups, dpss_vectors, sky_model_projected, gains):
    ants_map = {ant: i for i, ant in enumerate(gains.ant_array)}
    fg_comps_tensor = calamity.sparse_tensorize_pbl_fg_model_comps(
        red_grps=redundant_groups,
        fg_model_comps=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
    )
    (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        dpss_vectors,
        redundant_groups,
        time_index=0,
        polarization="xx",
        force2d=True,
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
    model_r = calamity.yield_fg_pbl_model_sparse_tensor(fg_comps_tensor, fg_coeffs_re, nants, nfreqs)
    model_i = calamity.yield_fg_pbl_model_sparse_tensor(fg_comps_tensor, fg_coeffs_im, nants, nfreqs)
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


def test_tensorize_pbl_data_dictionary(sky_model_projected, redundant_groups):
    for tnum in range(sky_model_projected.Ntimes):
        data_re, data_im, wgts = calamity.tensorize_pbl_data_dictionary(
            sky_model_projected, "xx", tnum, redundant_groups, dtype=np.float64
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                data = sky_model_projected.get_data(ap + ("xx",))[tnum]
                assert np.allclose(data, data_re[ap].numpy() + 1j * data_im[ap].numpy())
                assert np.allclose(wgts[ap], 1.0)


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
                assert np.allclose(wgts[i, j], 1.0)


def test_yield_data_model_pbl_dictionary(
    sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized
):
    corrupted = utils.apply_gains(sky_model_projected, gains_antscale_randomized, inverse=True)
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    for tnum in range(sky_model_projected.Ntimes):
        gains_re, gains_im = calamity.tensorize_gains(gains_antscale_randomized, "xx", tnum, dtype=np.float64)
        (
            fg_range_map,
            comp_tensor_map,
        ) = calamity.tensorize_pbl_model_comps_dictionary(redundant_groups, dpss_vectors, dtype=np.float64)
        (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            dpss_vectors,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            dtype=np.float64,
        )
        data_re, data_im, wgts = calamity.tensorize_pbl_data_dictionary(
            corrupted, "xx", tnum, redundant_groups, dtype=np.float64
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                model_r, model_i = calamity.yield_data_model_pbl_dictionary(
                    ap[0],
                    ap[1],
                    gains_re,
                    gains_im,
                    ants_map,
                    fg_coeffs_re,
                    fg_coeffs_im,
                    fg_range_map,
                    comp_tensor_map,
                )
                assert np.allclose(model_r.numpy(), data_re[ap].numpy())
                assert np.allclose(model_i.numpy(), data_im[ap].numpy())


def test_yield_data_model_pbl_sparse_tensor(
    sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized
):
    corrupted = utils.apply_gains(sky_model_projected, gains_antscale_randomized, inverse=True)
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    fg_comp_tensor = calamity.sparse_tensorize_pbl_fg_model_comps(
        red_grps=redundant_groups,
        fg_model_comps=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
    )
    nants = corrupted.Nants_data
    nfreqs = corrupted.Nfreqs
    for tnum in range(sky_model_projected.Ntimes):
        gains_re, gains_im = calamity.tensorize_gains(gains_antscale_randomized, "xx", tnum, dtype=np.float64)
        (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            dpss_vectors,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            force2d=True,
            dtype=np.float64,
        )
        data_re, data_im, wgts = calamity.tensorize_data(
            corrupted, redundant_groups, ants_map, "xx", tnum, dtype=np.float64
        )
        model_r, model_i = calamity.yield_data_model_sparse_tensor(
            gains_re,
            gains_im,
            fg_comp_tensor,
            fg_coeffs_re,
            fg_coeffs_im,
            nants,
            nfreqs,
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                i, j = ants_map[ap[0]], ants_map[ap[1]]
                assert np.allclose(model_r[i, j].numpy(), data_re[ap].numpy())
                assert np.allclose(model_i[i, j].numpy(), data_im[ap].numpy())


def test_cal_loss_sparse_tensor(sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized):
    corrupted = utils.apply_gains(sky_model_projected, gains_antscale_randomized, inverse=True)
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    fg_comp_tensor = calamity.sparse_tensorize_pbl_fg_model_comps(
        red_grps=redundant_groups,
        fg_model_comps=dpss_vectors,
        ants_map=ants_map,
        dtype=np.float64,
    )
    nants = corrupted.Nants_data
    nfreqs = corrupted.Nfreqs
    for tnum in range(sky_model_projected.Ntimes):
        gains_re, gains_im = calamity.tensorize_gains(gains_antscale_randomized, "xx", tnum, dtype=np.float64)
        (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            dpss_vectors,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            force2d=True,
            dtype=np.float64,
        )
        data_re, data_im, wgts = calamity.tensorize_data(
            corrupted, redundant_groups, ants_map, "xx", tnum, dtype=np.float64
        )
        model_r, model_i = calamity.yield_data_model_sparse_tensor(
            gains_re,
            gains_im,
            fg_comp_tensor,
            fg_coeffs_re,
            fg_coeffs_im,
            nants,
            nfreqs,
        )
        cal_loss = calamity.cal_loss_sparse_tensor(
            data_re,
            data_im,
            wgts,
            gains_re,
            gains_im,
            fg_comp_tensor,
            fg_coeffs_re,
            fg_coeffs_im,
            nants,
            nfreqs,
        ).numpy()

        assert np.isclose(cal_loss, 0.0)


def test_cal_loss_dictionary(sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized):
    corrupted = utils.apply_gains(sky_model_projected, gains_antscale_randomized, inverse=True)
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    (
        fg_range_map,
        comp_tensor_map,
    ) = calamity.tensorize_pbl_model_comps_dictionary(redundant_groups, dpss_vectors, dtype=np.float64)
    for tnum in range(sky_model_projected.Ntimes):
        gains_re, gains_im = calamity.tensorize_gains(gains_antscale_randomized, "xx", tnum, dtype=np.float64)
        (fg_coeffs_re, fg_coeffs_im,) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            dpss_vectors,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            dtype=np.float64,
        )
        data_re, data_im, wgts = calamity.tensorize_pbl_data_dictionary(
            corrupted, "xx", tnum, redundant_groups, dtype=np.float64
        )

        cal_loss = calamity.cal_loss_dictionary(
            gains_re,
            gains_im,
            fg_coeffs_re,
            fg_coeffs_im,
            data_re,
            data_im,
            wgts,
            ants_map,
            fg_range_map,
            comp_tensor_map,
        ).numpy()
        assert np.isclose(cal_loss, 0.0)


@pytest.mark.parametrize("method", ["sparse_tensor", "dictionary"])
def test_calibrate_and_model_dpss(uvdata, sky_model_projected, gains_randomized, method):
    for use_redundancy in [True, False]:
        # check that resid is much smaller then model and original data.
        model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=uvdata,
            gains=gains_randomized,
            verbose=True,
            use_redundancy=use_redundancy,
            sky_model=sky_model_projected,
            maxsteps=1000,
            modeling=method,
            correct_resid=True,
            correct_model=True,
        )
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert len(fit_history) == 1
        assert len(fit_history[0]) == 1

        # test that calibrating with a perfect sky model and only optimizing gains yields nearly perfect solutions for the gains.
        model, resid, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=sky_model_projected,
            gains=gains_randomized,
            verbose=True,
            use_redundancy=use_redundancy,
            sky_model=sky_model_projected,
            freeze_model=True,
            maxsteps=1000,
            modeling=method,
            correct_resid=True,
            correct_model=True,
        )
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert np.allclose(
            model.data_array,
            sky_model_projected.data_array,
            atol=1e-5 * np.mean(np.abs(model.data_array) ** 2.0) ** 0.5,
        )
        assert np.allclose(gains.gain_array, gains_randomized.gain_array, rtol=0.0, atol=1e-4)
        assert len(fit_history) == 1
        assert len(fit_history[0]) == 1
