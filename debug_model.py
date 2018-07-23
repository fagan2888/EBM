#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=1.00)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

# model.insolation(insolation_type='perturbation', perturb_center=15, perturb_spread=4.94, perturb_intensity=18)
model.insolation(insolation_type='annual_mean_clark')
# model.insolation(insolation_type='annual_mean')
# model.insolation(insolation_type='summer_mean')

model.albedo(albedo_feedback=False)

model.outgoing_longwave('full_wvf')
# model.outgoing_longwave('full_no_wvf')

model.solve(numerical_method='crank', nPlot=100, nPrint=100)

model.save_plots()
