from ..data import DATA_PATH
import pytest
from .. import calamity
from pyuvdata import UVData
from pyuvdata import UVCal
from .. import utils
import os
import numpy as np
import copy


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
    gains.gain_array += 1e-2 * np.random.randn(
        *gains.gain_array.shape
    ) + 1e-2j * np.random.randn(*gains.gain_array.shape)
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
            dpss_vectors[ap]
            @ (sky_model.data_array[dinds, 0, :, 0] @ dpss_vectors[ap]).T
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
    gains_r, gains_i = calamity.tensorize_gains(
        gains_antscale, polarization="xx", time_index=0, dtype=np.float64
    )
    assert gains_r.dtype == np.float64
    assert gains_i.dtype == np.float64
    for i, ant in enumerate(gains_antscale.ant_array):
        assert np.allclose(gains_r.numpy()[ant], ant + 1)
        assert np.allclose(gains_i.numpy()[ant], 0.0)


def test_tensorize_per_baseline_model_comps_dictionary(
    sky_model_projected, dpss_vectors, redundant_groups
):
    (
        foreground_range_map,
        component_tensor_map,
    ) = calamity.tensorize_per_baseline_model_comps_dictionary(
        redundant_groups, dpss_vectors, dtype=np.float64
    )
    for grp in redundant_groups:
        for ap in grp:
            tdata = sky_model_projected.get_data(ap[0], ap[1], "xx").T
            comps = component_tensor_map[ap].numpy()
            model = comps @ (tdata.T @ comps).T
            # check that dpss compoments accurately describe sky-model data to
            # the 1e-5 level.
            rmsdata = np.mean(np.abs(tdata) ** 2.0) ** 0.5
            assert np.allclose(model, tdata, rtol=0.0, atol=1e-5 * rmsdata)


def test_yield_fg_model_and_fg_coeffs(
    dpss_vectors, redundant_groups, sky_model_projected
):
    (
        foreground_range_map,
        component_tensor_map,
    ) = calamity.tensorize_per_baseline_model_comps_dictionary(
        redundant_groups, dpss_vectors, dtype=np.float64
    )
    (foreground_coeffs_real, foreground_coeffs_imag,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        component_tensor_map,
        redundant_groups,
        time_index=0,
        polarization="xx",
        dtype=np.float64,
    )
    # test that all model predictions are close to the underlying data they are derived from.
    for ap in component_tensor_map:
        (model_r, model_i,) = calamity.yield_fg_model_per_baseline_dictionary_method(
            ap[0],
            ap[1],
            foreground_coeffs_real,
            foreground_coeffs_imag,
            foreground_range_map,
            component_tensor_map,
        )
        model = model_r.numpy() + 1j * model_i.numpy()
        data = sky_model_projected.get_data(ap + ("xx",))
        rmsdata = np.mean(np.abs(data) ** 2.0) ** 0.5

        assert np.allclose(model, data, rtol=0.0, atol=1e-5 * rmsdata)


def test_insert_model_into_uvdata_dictionary(
    dpss_vectors, redundant_groups, sky_model_projected
):
    (
        foreground_range_map,
        component_tensor_map,
    ) = calamity.tensorize_per_baseline_model_comps_dictionary(
        redundant_groups, dpss_vectors, dtype=np.float64
    )
    (foreground_coeffs_real, foreground_coeffs_imag,) = calamity.tensorize_fg_coeffs(
        sky_model_projected,
        component_tensor_map,
        redundant_groups,
        time_index=0,
        polarization="xx",
    )
    inserted_model = copy.deepcopy(sky_model_projected)
    # set data array to be noise.
    inserted_model.data_array = np.random.randn(
        *inserted_model.data_array.shape
    ) + 1j * np.random.randn(*inserted_model.data_array.shape)
    # inserte tensors
    calamity.insert_model_into_uvdata_dictionary(
        inserted_model,
        0,
        "xx",
        foreground_range_map,
        component_tensor_map,
        foreground_coeffs_real,
        foreground_coeffs_imag,
    )
    # check that data arrays are equal
    assert np.allclose(inserted_model.data_array, sky_model_projected.data_array)


def test_tensorize_per_baseline_data_dictionary(sky_model_projected, redundant_groups):
    for tnum in range(sky_model_projected.Ntimes):
        data_real, data_imag, wgts = calamity.tensorize_per_baseline_data_dictionary(
            sky_model_projected, "xx", tnum, redundant_groups, dtype=np.float64
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                data = sky_model_projected.get_data(ap + ("xx",))[tnum]
                assert np.allclose(
                    data, data_real[ap].numpy() + 1j * data_imag[ap].numpy()
                )
                assert np.allclose(wgts[ap], 1.0)


def test_yield_data_model_per_baseline_dictionary(
    sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized
):
    calibrated = utils.apply_gains(
        sky_model_projected, gains_antscale_randomized, inverse=True
    )
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    for tnum in range(sky_model_projected.Ntimes):
        gains_real, gains_imag = calamity.tensorize_gains(
            gains_antscale_randomized, "xx", tnum, dtype=np.float64
        )
        (
            foreground_range_map,
            component_tensor_map,
        ) = calamity.tensorize_per_baseline_model_comps_dictionary(
            redundant_groups, dpss_vectors, dtype=np.float64
        )
        (
            foreground_coeffs_real,
            foreground_coeffs_imag,
        ) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            component_tensor_map,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            dtype=np.float64,
        )
        data_real, data_imag, wgts = calamity.tensorize_per_baseline_data_dictionary(
            calibrated, "xx", tnum, redundant_groups, dtype=np.float64
        )
        for red_grp in redundant_groups:
            for ap in red_grp:
                model_r, model_i = calamity.yield_data_model_per_baseline_dictionary(
                    ap[0],
                    ap[1],
                    gains_real,
                    gains_imag,
                    ants_map,
                    foreground_coeffs_real,
                    foreground_coeffs_imag,
                    foreground_range_map,
                    component_tensor_map,
                )
                assert np.allclose(model_r.numpy(), data_real[ap].numpy())
                assert np.allclose(model_i.numpy(), data_imag[ap].numpy())


def test_cal_loss_dictionary(
    sky_model_projected, dpss_vectors, redundant_groups, gains_antscale_randomized
):
    calibrated = utils.apply_gains(
        sky_model_projected, gains_antscale_randomized, inverse=True
    )
    ants_map = {ant: i for i, ant in enumerate(gains_antscale_randomized.ant_array)}
    for tnum in range(sky_model_projected.Ntimes):
        gains_real, gains_imag = calamity.tensorize_gains(
            gains_antscale_randomized, "xx", tnum, dtype=np.float64
        )
        (
            foreground_range_map,
            component_tensor_map,
        ) = calamity.tensorize_per_baseline_model_comps_dictionary(
            redundant_groups, dpss_vectors, dtype=np.float64
        )
        (
            foreground_coeffs_real,
            foreground_coeffs_imag,
        ) = calamity.tensorize_fg_coeffs(
            sky_model_projected,
            component_tensor_map,
            redundant_groups,
            time_index=tnum,
            polarization="xx",
            dtype=np.float64,
        )
        data_real, data_imag, wgts = calamity.tensorize_per_baseline_data_dictionary(
            calibrated, "xx", tnum, redundant_groups, dtype=np.float64
        )

        cal_loss = calamity.cal_loss_dictionary(
            gains_real,
            gains_imag,
            foreground_coeffs_real,
            foreground_coeffs_imag,
            data_real,
            data_imag,
            wgts,
            ants_map,
            foreground_range_map,
            component_tensor_map,
        ).numpy()
        assert np.isclose(cal_loss, 0.0)


def test_calibrate_and_model_dpss(uvdata, sky_model_projected, gains_randomized):
    for use_redundancy in [True, False]:
        # check that resid is much smaller then model and original data.
        model, resid, filtered, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=uvdata,
            gains=gains_randomized,
            verbose=True,
            use_redundancy=use_redundancy,
            sky_model=sky_model_projected,
            maxsteps=3000,
        )
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert len(loss_history) == 1
        assert len(fit_history[0]) == 1

        # test that calibrating with a perfect sky model and only optimizing gains yields nearly perfect solutions for the gains.
        model, resid, filtered, gains, fit_history = calamity.calibrate_and_model_dpss(
            min_dly=2.0 / 0.3,
            offset=2.0 / 0.3,
            uvdata=sky_model_projected,
            gains=gains_randomized,
            verbose=True,
            use_redundancy=use_redundancy,
            sky_model=sky_model_projected,
            freeze_model=True,
            maxsteps=3000,
        )
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.0)) >= 1e3 * np.sqrt(
            np.mean(np.abs(resid.data_array) ** 2.0)
        )
        assert np.allclose(
            model.data_array,
            sky_model_projected.data_array,
            atol=1e-5 * np.mean(np.abs(model.data_array) ** 2.0) ** 0.5,
        )
        assert np.allclose(
            gains.gain_array, gains_randomized.gain_array, rtol=0.0, atol=1e-4
        )
        assert len(loss_history) == 1
        assert len(fit_history[0]) == 1
