#!/usr/bin/env python

from EBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=, dtmax_multiple=, max_sim_years=, tol=, diffusivity=)

model.initial_temperature(initial_condition=, low=, high=)

model.insolation(insolation_type=, perturb_center=, perturb_spread=, perturb_intensity=)
        
model.albedo(al_feedback=, alb_ice=, alb_water=)

model.outgoing_longwave(olr_type=, A=, B=, emissivity=)

model.solve(numerical_method=, frames=)

model.save_data(control=)

model.log_efe(fname_efe=)

model.log_feedbacks(fname_feedbacks=)

model.save_plots()
