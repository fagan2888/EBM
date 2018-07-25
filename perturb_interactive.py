#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.001)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

model.albedo(albedo_feedback=False)

model.outgoing_longwave('full_wvf', RH_vert_profile='steps', RH_lat_profile='gaussian')

# sigmas = [4.94, 9.89]
# lats   = [15, 60]
# fname = 'perturbed_efe_planck_linear_fit.dat'
# for sigma, lat0 in zip(sigmas, lats):
#     for M in [5, 10, 15, 18]:
#         model.insolation(insolation_type='perturbation', perturb_center=lat0, perturb_spread=sigma, perturb_intensity=M)
#         model.solve(numerical_method='crank', nPlot=100, nPrint=500)
#         model.log_efe(fname)

lat0 = 15; sigma = 4.94; M = 15
model.insolation(insolation_type='perturbation', perturb_center=lat0, perturb_spread=sigma, perturb_intensity=M)
model.solve(numerical_method='crank', nPlot=100, nPrint=500)
model.save_data()
model.log_efe('efe_full_wvf_m15.txt')
