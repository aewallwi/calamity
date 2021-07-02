from ..data import DATA_PATH
from .. import modeling
import pytest
from pyuvdata import UVData
import os


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


def test_get_uv_overlapping_grps_conjugated(sky_model):
    fitting_grps, fitting_vec_centers, connections, grp_labels = modeling.get_uv_overlapping_grps_conjugated(
        uvdata=sky_model, red_tol_freq=0.5, n_angle_bins=200
    )
    assert fitting_grps == [
        [((0, 1),)],
        [((3, 4),)],
        [((1, 2),)],
        [((0, 2),)],
        [((4, 5),)],
        [((2, 3),), ((3, 5),), ((2, 4),), ((1, 3),), ((0, 3),), ((1, 4),), ((0, 4),), ((2, 5),)],
        [((1, 5),), ((0, 5),)],
    ]
