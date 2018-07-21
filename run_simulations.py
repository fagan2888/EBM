#!/usr/bin/env python

from DEBM import Model

sim = Model(dlat=0.5, dtmax_multiple=1.0, max_iters=1e5, tol=0.01)

sim.initial_temperature(initial_condition='triangle', triangle_low=270, triangle_high=305)

# sim.insolation(insolation_type='perturbation', perturb_center=15, perturb_spread=4.94, perturb_intensity=18)
sim.insolation(insolation_type='annual_mean_clark')
# sim.insolation(insolation_type='annual_mean')
# sim.insolation(insolation_type='summer_mean')

sim.albedo(albedo_feedback=False)

## Fitted values from a full_wvf simulation
#A_full_wvf    = -417.0478973801873
#B_full_wvf    = 2.349441658553002
## A reference temp to calculate A given B
#T_ref         = 275
## Feedback parameters from SoldenHeld2003
#lambda_planck = -3.10 #W/m2/K
#lambda_water  = +1.80
#lambda_clouds = +0.68
#lambda_albedo = +0.26
#lambda_lapse  = -0.84
## A few tests we want to do:
## just planck
#B = -(lambda_planck)
## planck + water + lapse = full_wvf
##B = -(lambda_planck + lambda_water + lambda_lapse)
## planck + lapse = full_no_wvf
##B = -(lambda_planck + lambda_lapse)
## Calculate a decent Acoeff:
#A = A_full_wvf + B_full_wvf * T_ref - B * T_ref

# sim.outgoing_longwave('linear', A=A, B=B)
# sim.outgoing_longwave('planck', emissivity=0.6)
sim.outgoing_longwave('full_wvf')

sim.solve(numerical_method='crank', nPlot=1000, nPrint=1000)

sim.save_data()

#root = sim.calculate_efe()
#print(root)

sim.save_plots()

