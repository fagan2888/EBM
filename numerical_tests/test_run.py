#!/usr/bin/env python

from EBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=401, dtmax_multiple=0.5, max_sim_years=5.0, tol=1e-10)
# model = EnergyBalanceModel(N_pts=401, dtmax_multiple=100.0, max_sim_years=5.0, tol=1e-9)

# model.initial_temperature(initial_condition='load_data', low=None, high=None)
model.initial_temperature(initial_condition='legendre', low=270, high=310)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)

model.insolation(insolation_type='perturbation', perturb_center=15, perturb_spread=4.94, perturb_intensity=10)
        
model.outgoing_longwave(olr_type='planck', A=-572.3, B=2.92, emissivity=0.65, RH_vert_profile='steps', RH_lat_profile='mid_level_gaussian', gaussian_spread=5)

model.solve(numerical_method='explicit', frames=100)

model.save_data()

model.log_efe(fname_efe='itcz.log')

# model.log_feedbacks(fname_feedbacks='feedbacks.log')

model.save_plots()
