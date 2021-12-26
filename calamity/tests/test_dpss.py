import pytest
from pyuvdata import UVData
import os
from .. import dpss
import scipy.signal.windows as windows


@pytest.mark.parametrize(
    "use_tensorflow",
    "tf_method",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dpss_operator(use_tensorflow, tf_method):
    # test that an error is thrown when we specify more then one
    # termination method.
    NF = 100
    DF = 100e3
    freqs = np.arange(-NF / 2, NF / 2) * DF + 150e6
    freqs_bad = np.array([1.100912386458, 1.22317, 2.12341260, 3.234632462, 5.32348356887])
    # now calculate DPSS operator matrices using different cutoff criteria. The columns
    # should be the same up to the minimum number of columns of the three techniques.
    amat1, ncol1 = dpss.dpss_operator(
        freqs,
        [0.0],
        [100e-9],
        eigenval_cutoff=[1e-9],
        use_tensorflow=use_tensorflow,
        lobe_ordering_like_scipy=True,
        tf_method=tf_method,
    )
    dpss_mat = windows.dpss(NF, NF * DF * 100e-9, ncol1[0]).T
    for m in range(ncol1[0]):
        # deal with -1 degeneracy which can come up since vectors are obtained from spectral decomposition.
        # and there is degeneracy between +/-1 eigenval and eigenvector.
        # This simply effects the sign of the vector to be fitted
        # and since fitting will cancel by obtaining a -1 on the coefficient, this sign is meaningless
        assert np.allclose(amat1[:, m], dpss_mat[:, m]) or np.allclose(amat1[:, m], -dpss_mat[:, m])

    nf = 100
    times = np.linspace(-1800, 1800.0, nf, endpoint=False)
    dt = times[1] - times[0]
    tarr = np.arange(-nf / 2, nf / 2) * dt
    filter_centers = [0.0]
    filter_half_widths = [0.004]
    amat1, ncol1 = dspec.dpss_operator(
        tarr,
        [0.0],
        filter_half_widths,
        eigenval_cutoff=[1e-9],
        use_tensorflow=use_tensorflow,
        lobe_ordering_like_scipy=True,
        tf_method=tf_method,
    )
    dpss_mat = windows.dpss(nf, nf * dt * filter_half_widths[0], ncol1[0]).T
    for m in range(ncol1[0]):
        assert np.allclose(amat1[:, m], dpss_mat[:, m]) or np.allclose(amat1[:, m], -dpss_mat[:, m])
