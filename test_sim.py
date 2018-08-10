#!/usr/bin/env python

from DEBM import Model

# model = Model(dlat=0.5, dtmax_multiple=0.1, max_sim_years=5, tol=1e-10)
# model = Model(dlat=0.5, dtmax_multiple=1.50, max_sim_years=5, tol=1e-8)
model = Model(dlat=0.5, dtmax_multiple=10.0, max_sim_years=5, tol=1e-7)

# model.initial_temperature(initial_condition='load_data', low=None, high=None)
model.initial_temperature(initial_condition='legendre', low=270, high=305)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)


model.insolation(insolation_type='perturbation', perturb_center=15, 
        perturb_spread=4.94, perturb_intensity=10)
# model.insolation(insolation_type='perturbation', perturb_center=60, 
#         perturb_spread=9.89, perturb_intensity=18)
        
model.outgoing_longwave(olr_type='full_wvf', A=271.8, B=0.0, emissivity=0.6, 
        RH_vert_profile='steps', RH_lat_profile='mid_level_gaussian', gaussian_spread1=5, 
        gaussian_spread2=None, scale_efe=False, constant_spec_hum=False)

# model.solve(numerical_method='explicit', frames=100)
model.solve(numerical_method='implicit', frames=250)

model.save_data()

if model.insolation_type == 'perturbation':
    model.log_efe(fname='itcz.log')

model.log_feedbacks(fname='feedbacks.log')

model.save_plots()
