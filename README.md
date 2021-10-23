[![Python Package using Conda](https://github.com/aewallwi/calamity/actions/workflows/ci.yml/badge.svg)](https://github.com/aewallwi/calamity/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aewallwi/calamity/branch/main/graph/badge.svg?token=CoPpmdXRmF)](https://codecov.io/gh/aewallwi/calamity)
# CALAMITY -- A Radio Interferometer Calibration Tool that Requires No Redundancy and Minimal Prior Knowledge on Foregrounds and Beams

## Summary
<span style="font-variant:small-caps;">calamity</span> is a frequency regulated self-cal strategy that simultaneously fits for calibration solutions with foregrounds that are described by a well understood / behaved set of  basis vectors. Examples of basis sets that do not incorporate inter-baseline correlations are DPSS vectors and DFT vectors. <span style="font-variant:small-caps;">calamity</span> can also handle fitting modes with inter-baseline correlations to reduce sample variance and potentially recover modes inside of the wedge.


## Installation. 
You can install `calamity` with `pip` by running `pip install git+https://github.com/aewallwi/calamity.git` in your terminal.
