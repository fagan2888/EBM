#!/usr/bin/env python

from MEBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=, dtmax_multiple=, max_sim_years=, tol=, diffusivity=)

model.set_init_temp(init_temp_type=, low=, high=)

model.set_insol(insol_type=, perturb_center=, perturb_spread=, perturb_intensity=)
        
model.set_albedo(al_feedback=, alb_ice=, alb_water=)

model.set_olr(olr_type=, A=, B=, emissivity=)

model.solve(numerical_method=, frames=)

model.save_data(control=)

model.log_efe(fname_efe=)

model.log_feedbacks(fname_feedbacks=)

model.save_plots()
