#!/usr/bin/env python
# bash interface script for dpss calibration and modeling.


from calamity import calibration

ap = calibration.dpss_fit_argparser()
args = ap.parse_args()
args.correct_model = not(args.dont_correct_model)
del args.dont_correct_model
calibration.read_calibrate_and_model_dpss(**vars(args))
