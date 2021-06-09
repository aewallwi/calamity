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
    uvd.read_uvh5(os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_gsm.uvh5'))
    return uvd


@pytest.fixture
def uvdata(sky_model):
    uvd = UVData()
    uvd.read_uvh5(os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_eor_-50.0dB.uvh5'))
    uvd.data_array = uvd.data_array + sky_model.data_array
    return uvd


@pytest.fixture
def gains(sky_model):
    return utils.blank_uvcal_from_uvdata(sky_model)


@pytest.fixture
def gains_antscale(gains):
    for i, antnum in enumerate(gains.ant_array):
        gains.gain_array[i] *= (antnum + 1.)
    return gains


@pytest.fixture
def dpss_vectors(sky_model):
    return utils.yield_dpss_model_components(sky_model)


@pytest.fixture
def redundant_groups(sky_model):
    return  calamity.get_redundant_groups_conjugated(sky_model)


def test_renormalize(sky_model, gains):
    gains.gain_array *= (51. + 23j) ** -.5
    sky_model_ref = copy.deepcopy(sky_model)
    sky_model.data_array *= 51. + 23j
    assert not np.allclose(gains.gain_array, 1.)
    assert not np.allclose(sky_model_ref.data_array, sky_model.data_array)
    calamity.renormalize(sky_model_ref, sky_model, gains, polarization='xx')
    assert np.allclose(gains.gain_array, 1.)
    assert np.allclose(sky_model_ref.data_array, sky_model.data_array)

def test_tensorize_gains(gains_antscale):
    gains_r, gains_i = calamity.tensorize_gains(gains_antscale, polarization='xx', time_index=0, dtype=np.float64)
    assert gains_r.dtype == np.float64
    assert gains_i.dtype == np.float64
    for i, ant in enumerate(gains_antscale.ant_array):
        assert np.allclose(gains_r.numpy()[ant], ant + 1)
        assert np.allclose(gains_i.numpy()[ant], 0.)

def test_tensorize_per_baseline_model_components_dictionary(sky_model, dpss_vectors, redundant_groups):
    foreground_range_map, component_tensor_map = calamity.tensorize_per_baseline_model_components_dictionary(redundant_groups, foreground_modeling_components, offset=2. / .3, min_dly = 2. / .3,
                                                                                                             dtype=np.float64)
    rmsdata = np.mean(np.abs(sky_model.data_array) ** 2.) ** .5
    for grp in redundant_groups:
        for ap in grp:
            tdata = sky_model.get_data(ap[0], ap[1], 'xx')
            comps = component_tensor_map[ap].numpy()
            model = (tdata @ comps).numpy()
            # check that dpss compoments accurately describe sky-model data to
            # the 1e-5 level.
            assert np.allclose(model, tdata, rtol=0., atol=1e-5 * rmsdata)


def test_yield_foreground_model_and_foreground_coeffs(dpss_vectors, redundant_groups, sky_model):
    rmsdata = np.mean(np.abs(sky_model.data_array) ** 2.) ** .5
    foreground_range_map, component_tensor_map = calamity.tensorize_per_baseline_model_components_dictionary(redundant_groups, foreground_modeling_components,
                                                                                                             offset=2. / .3, min_dly = 2. / .3, dtype=np.float64)
    foreground_coefficients_real, foreground_coefficients_imag = calamity.tensorize_foreground_coeffs(sky_model, component_tensor_map, redundant_groups, time_index=0,
                                                                                                      polarization='xx', dtype=np.float64)
    # test that all model predictions are close to the underlying data they are derived from.
    for ap in component_tensor_map:
        model_r, model_i = calamity.yield_foreground_model_per_baseline_dictionary_method(ap[0], ap[1], foreground_coeffs_real, foreground_coeffs_imag,
                                                                                          foreground_range_map, component_tensor_map)
        model = model_r.numpy() + 1j * model_i.numpy()
        data = sky_model.get_data(ap + ('xx', ))
        assert np.allclose(model, data, rtol=0., atol=1e-5 * rmsdata)

#def test_insert_model_into_uvdata_dictionary(dpss_vectors, redundant_groups, sky_model):


def test_calibrate_and_model_dpss(uvdata, sky_model, gains):
    for use_redundancy in [True, False]:
        # check that resid is much smaller then model and original data.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=uvdata, gains=gains, verbose=True,
                                                                                        use_redundancy=use_redundancy, maxsteps=300)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert len(fitting_info) == 1
        assert len(fitting_info[0]) == 1

        # test that calibrating with a perfect sky model and only optimizing gains yields gains that are nearly unity.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=uvdata, gains=gains, verbose=True,
                                                                                        use_redundancy=use_redundancy, sky_model=sky_model,
                                                                                        freeze_model=True, maxsteps=300)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.allclose(gains.gain_array, 1., rtol=0., atol=1e-4)
        assert len(fitting_info) == 1
        assert len(fitting_info[0]) == 1
