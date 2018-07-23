#!/usr/bin/env python

from DEBM import Model

model = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.001)

model.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

model.albedo(albedo_feedback=False)

# Fitted values from a full_wvf modelulation
A_full_wvf    = -417.0478973801873
B_full_wvf    = 2.349441658553002
# A reference temp to calculate A given B
T_ref         = 275
# Feedback parameters from SoldenHeld2003
lambda_planck = -3.10 #W/m2/K
lambda_water  = +1.80
lambda_clouds = +0.68
lambda_albedo = +0.26
lambda_lapse  = -0.84
# A few tests we want to do:

# Just planck
# B = -(lambda_planck)

# Planck + water + lapse = full_wvf
# B = -(lambda_planck + lambda_water + lambda_lapse)

# Planck + lapse = full_no_wvf
B = -(lambda_planck + lambda_lapse)

# Calculate a decent A:
A = A_full_wvf + B_full_wvf * T_ref - B * T_ref

# model.outgoing_longwave('linear', A=A, B=B)
# model.outgoing_longwave('planck', emissivity=0.6)
model.outgoing_longwave('full_wvf')
# model.outgoing_longwave('full_no_wvf')

model.insolation(insolation_type='annual_mean_clark')

model.solve(numerical_method='crank', nPlot=100, nPrint=500)

model.save_data()
