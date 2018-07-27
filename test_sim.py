#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.1)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

model.albedo(albedo_feedback=False, alb_ice=None, alb_water=None)

model.insolation(insolation_type='annual_mean_clark', perturb_center=None, perturb_spread=None, perturb_intensity=None)

model.outgoing_longwave(olr_type='full_wvf', A=None, B=None, emissivity=None, RH_vert_profile='steps', RH_lat_profile='gaussian', scale_efe=False)

# model.solve(numerical_method='crank', nPlot=100, nPrint=500)

# model.save_data()

# model.log_efe(fname='efe.log')

# model.save_plots()
