from ..data import DATA_PATH
import pytest
from .. import calamity
from pyuvdata import UVData
from pyuvdata import UVCal
from .. import utils
import os
import numpy as np

class TestCalamity():
    def setup(self):
        self.uvdata = UVData()
        self.uvdata.read_uvh5(os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_eor_-50.0dB.uvh5'))
        self.sky_model = UVData()
        self.sky_model.read_uvh5(os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_gsm.uvh5'))
        self.uvdata.data_array += self.sky_model.data_array
        self.fgfile = os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_eor_-50.0dB.uvh5')
        self.eorfile = os.path.join(DATA_PATH, 'Garray_antenna_diameter2.0_fractional_spacing1.0_nant6_nf200_df100.000kHz_f0100.000MHzcompressed_True_autosFalse_gsm.uvh5')
        self.gains = utils.blank_uvcal_from_uvdata(self.uvdata)

    def teardown(self):
        del self.uvdata
        del self.sky_model
        del self.gains
        del self.eorfile
        del self.fgfile

    def test_calibrate_and_model_dpss(self):
        self.setup()
        # check that resid is much smaller then model and original data.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=self.uvdata, gains=self.gains, maxsteps=3000)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.sqrt(np.mean(np.abs(self.uvdata.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))

        # test that calibrating with a perfect sky model and only optimizing gains yields gains that are nearly unity.
        model, resid, filtered, gains, fitting_info = calamity.calibrate_and_model_dpss(min_dly=2.0/.3, uvdata=self.sky_model, gains=self.gains, sky_model=self.sky_model, freeze_model=True, maxsteps=3000)
        assert np.sqrt(np.mean(np.abs(model.data_array) ** 2.)) >= 1e3 * np.sqrt(np.mean(np.abs(resid.data_array) ** 2.))
        assert np.allclose(gains.gain_array, 1., rtol=0., atol=1e-4)
        self.teardown()

    #def test
