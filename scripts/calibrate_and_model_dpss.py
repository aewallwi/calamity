#!/usr/bin/env python
# bash interface script for dpss calibration and modeling.


from calamity import calibration

ap = calibration.dpss_fit_argparser()
args = ap.parse_args()
calibration.read_calibrate_and_model_dpss(**vars(args))
