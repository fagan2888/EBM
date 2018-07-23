#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.001)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

# model.insolation(insolation_type='perturbation', perturb_center=15, perturb_spread=4.94, perturb_intensity=18)
model.insolation(insolation_type='annual_mean_clark')
# model.insolation(insolation_type='annual_mean')
# model.insolation(insolation_type='summer_mean')

model.albedo(albedo_feedback=False)

A = -417.0478973801873
B = 2.349441658553002
model.outgoing_longwave('linear', A=A, B=B)
# model.outgoing_longwave('planck', emissivity=0.6)
# model.outgoing_longwave('full_wvf')

model.solve(numerical_method='crank', nPlot=1000, nPrint=1000)

model.save_data()

model.save_plots()
