#!/usr/bin/env python
# bash interface script for dpss calibration and modeling.


import calamity

ap = calamity.dpss_fit_argparser()
args = ap.parse_args()
calamity.calibrate_and_model_dpss(**vars(args))
