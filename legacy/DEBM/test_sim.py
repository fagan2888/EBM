#!/usr/bin/env python

from DEBM import Model

# model = Model(dlat=0.5, dtmax_multiple=0.1, max_sim_years=5, tol=1e-10)
# model = Model(dlat=0.5, dtmax_multiple=1.50, max_sim_years=5, tol=1e-8)
model = Model(dlat=0.5, dtmax_multiple=0.5, max_sim_years=5, tol=1e-8)

# model.initial_temperature(initial_condition='load_data', low=None, high=None)
model.initial_temperature(initial_condition='legendre', low=270, high=310)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)
# model.albedo(albedo_feedback=True, alb_ice=0.4, alb_water=0.1)


model.insolation(insolation_type='annual_mean_clark', perturb_center=15, 
# model.insolation(insolation_type='perturbation', perturb_center=15, 
        perturb_spread=4.94, perturb_intensity=10)
        
model.outgoing_longwave(olr_type='planck', A=271.8, B=1.8, emissivity=0.6, 
        RH_vert_profile='steps', RH_lat_profile='mid_level_gaussian', gaussian_spread1=5, 
        gaussian_spread2=None, scale_efe=False, constant_spec_hum=False)

# model.solve(numerical_method='explicit', frames=100)
# model.solve(numerical_method='implicit', frames=1000)
model.solve(numerical_method='implicit', frames=250)

# model.save_data()

# model.log_efe(fname_efe='itcz.log')

# model.log_feedbacks(fname_feedbacks='feedbacks.log')

model.save_plots()
