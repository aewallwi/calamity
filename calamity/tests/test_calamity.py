from ..data import DATA_PATH
import pytest
from .. import calamity
from pyuvdata import UVData
from pyuvdata import UVCal
from .. import utils
import os
import numpy as np

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

def test_calibrate_and_model_dpss(uvdata, sky_model, gains):
    for use_redundancy in [True, False]:
        # check that resid is much smaller then model and original data.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=uvdata, gains=gains, verbose=True,
                                                                                        use_redundancy=use_redundancy, maxsteps=300)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.sqrt(np.mean(np.abs(uvdata.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))

        # test that calibrating with a perfect sky model and only optimizing gains yields gains that are nearly unity.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=uvdata, gains=gains, verbose=True,
                                                                                        use_redundancy=use_redundancy, sky_model=sky_model,
                                                                                        freeze_model=True, maxsteps=300)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.allclose(gains.gain_array, 1., rtol=0., atol=1e-4)
        assert False
