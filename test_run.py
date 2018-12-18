#!/usr/bin/env python

from EBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=400, dtmax_multiple=0.5, max_sim_years=5.0, tol=1e-8)

# model.initial_temperature(initial_condition='load_data', low=None, high=None)
model.initial_temperature(initial_condition='legendre', low=270, high=300)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)
# model.albedo(albedo_feedback=True, alb_ice=0.4, alb_water=0.1)


model.insolation(insolation_type='perturbation', perturb_center=15, perturb_spread=4.94, perturb_intensity=0)
        
model.outgoing_longwave(olr_type='planck', A=300.0, B=0.0, emissivity=0.6, RH_vert_profile='steps', RH_lat_profile='mid_level_gaussian', gaussian_spread1=5, gaussian_spread2=None, scale_efe=False, constant_spec_hum=False)

# model.solve(numerical_method='explicit', frames=100)
model.solve(numerical_method='implicit', frames=250)

model.save_data()

model.log_efe(fname_efe='itcz.log')

# model.log_feedbacks(fname_feedbacks='feedbacks.log')

model.save_plots()
