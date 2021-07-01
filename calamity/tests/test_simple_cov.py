from ..data import DATA_PATH
from .. import simple_cov
import pytest
from pyuvdata import UVData
import os
import numpy as np


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


@pytest.mark.parametrize(
    "use_tensorflow, horizon, offset, min_dly, ant_dly",
    [(True, 1.0, 20.0, 0.0, 0.0), (False, 0.8, 123.0, 200.0, 0.0),
     (True, 1.0, 0.0, 0.0, 2 / .3)],
)
def test_simple_cov(use_tensorflow, horizon, offset, min_dly, sky_model, ant_dly):
    sky_model.select(bls=[(0, 1)])
    blvecs = sky_model.uvw_array
    freqs = sky_model.freq_array[0]
    nfreqs = len(freqs)
    fg0, fg1 = np.meshgrid(freqs, freqs)
    bldly = np.max([np.linalg.norm(blvecs[0]) * horizon / 3e8 + offset / 1e9, min_dly / 1e9])
    tcov = np.sinc(2 * bldly * (fg0 - fg1))
    if ant_dly > 0:
        tcov *= np.sinc(2 * (fg0 - fg1) * ant_dly)
    scov = simple_cov.simple_cov_matrix(
        blvecs,
        freqs,
        ant_dly=ant_dly,
        horizon=horizon,
        offset=offset,
        min_dly=min_dly,
        dtype=np.float64,
        use_tensorflow=use_tensorflow,
    )
    assert np.allclose(scov, tcov)
