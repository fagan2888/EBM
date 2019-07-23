#!/usr/bin/env python

from EBM import EnergyBalanceModel

# model = EnergyBalanceModel(N_pts=401, dtmax_multiple=1e3, max_sim_years=10.0, tol=1e-8, diffusivity="constant")
model = EnergyBalanceModel(N_pts=513, dtmax_multiple=200, max_sim_years=10.0, tol=1e-9, diffusivity="constant")

# model.initial_temperature(initial_condition="load_data", low=None, high=None)
model.initial_temperature(initial_condition="legendre", low=250, high=300)

model.insolation(insolation_type="perturbation", perturb_center=15, perturb_spread=4.94, perturb_intensity=0)
# model.insolation(insolation_type="perturbation", perturb_center=60, perturb_spread=9.89, perturb_intensity=5)
        
# model.albedo(al_feedback=False, alb_ice=0.6, alb_water=0.2)
model.albedo(al_feedback=True, alb_ice=0.6, alb_water=0.2)

model.outgoing_longwave(olr_type="planck", A=None, B=None, emissivity=0.65)
# model.outgoing_longwave(olr_type="full_radiation", A=None, B=None, emissivity=None)
# model.outgoing_longwave(olr_type="full_radiation_rh", A=None, B=None, emissivity=None)
# model.outgoing_longwave(olr_type="full_radiation_no_lr", A=None, B=None, emissivity=None)
# model.outgoing_longwave(olr_type="full_radiation_no_wv", A=None, B=None, emissivity=None)
# model.outgoing_longwave(olr_type="full_radiation_no_wv_no_lr", A=None, B=None, emissivity=None)
# model.outgoing_longwave(olr_type="full_radiation_homog", A=None, B=None, emissivity=None)

model.solve(numerical_method="implicit", frames=100)

model.save_data(control=False)

model.log_efe(fname_efe="itcz.log")

# model.log_feedbacks(fname_feedbacks="feedbacks.log")

# model.predict_efe()

# model.save_plots()
