#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.001)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

model.albedo(albedo_feedback=False)

# Fitted values from a full_wvf modelulation
A_full_wvf    = -412.05
B_full_wvf    =    2.33
# A reference temp to calculate A given B
T_ref         = 275
# Feedback parameters from SoldenHeld2003
lambda_planck = -3.10 #W/m2/K
lambda_water  = +1.80
lambda_clouds = +0.68
lambda_albedo = +0.26
lambda_lapse  = -0.84

# # Just planck
# B = -(lambda_planck)

# # Planck + water + lapse = full_wvf
# B = -(lambda_planck + lambda_water + lambda_lapse)

# # Planck + lapse = full_no_wvf
# B = -(lambda_planck + lambda_lapse)

# # Calculate a decent A:
# A = A_full_wvf + B_full_wvf * T_ref - B * T_ref
# print('A = {:3.2f}; B = {:1.2f}'.format(A, B))

# Values from SoldenHeld2003
# A = -623.80; B = 3.10    # planck
# A = -359.80; B = 2.14    # wvf
# A = -854.80; B = 3.94    # no wvf

# Fits:
# A = -652.88; B = 3.10    # planck
# A = -412.05; B = 2.33    # wvf
# A = -418.26; B = 2.36    # no wvf

# model.outgoing_longwave('linear', A=A, B=B)
# model.outgoing_longwave('planck', emissivity=0.6)
# model.outgoing_longwave('full_wvf')
model.outgoing_longwave('full_no_wvf', RH_profile='zero_top')

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
model.log_efe('efe_full_no_wvf_m15.txt')
