#!/usr/bin/env python

from EBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=, dtmax_multiple=, max_sim_years=, tol=)

model.initial_temperature(initial_condition=, low=, high=)

model.albedo(albedo_feedback=, alb_ice=, alb_water=)

model.insolation(insolation_type=, perturb_center=, perturb_spread=, perturb_intensity=)

model.outgoing_longwave(olr_type=, A=, B=, emissivity=, RH_vert_profile=, RH_lat_profile=, gaussian_spread=, constant_spec_hum=)

model.solve(numerical_method=, frames=)

model.save_data()

model.log_efe(fname_efe=)

# model.log_feedbacks(fname_feedbacks=)

model.save_plots()
