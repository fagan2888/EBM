#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=, dtmax_multiple=, max_iters=, tol=)

model.initial_temperature(initial_condition=, triangle_low=, triangle_high=)

model.albedo(albedo_feedback=, alb_ice=, alb_water=)

model.outgoing_longwave(olr_type=, A=, B=, emissivity=, RH_vert_profile=, RH_lat_profile=)

model.insolation(insolation_type=, perturb_center=, perturb_spread=, perturb_intensity=)

model.solve(numerical_method=, nPlot=, nPrint=)

model.save_data()

model.log_efe(fname=)

model.save_plots()
