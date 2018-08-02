#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=0.1, max_iters=1e5, tol=0.01)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)

model.insolation(insolation_type='annual_mean', perturb_center=None, perturb_spread=None, perturb_intensity=None)

model.outgoing_longwave(olr_type='linear', A=-450, B=2.5, emissivity=0.6, RH_vert_profile=None, RH_lat_profile=None, 
        gaussian_spread1=None, gaussian_spread2=None, scale_efe=None)

model.solve(numerical_method='euler_back', nPlot=100, nPrint=500)

model.save_data()

# model.log_efe(fname='efe.log')

model.save_plots()
