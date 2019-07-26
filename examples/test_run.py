#!/usr/bin/env python

import sys
sys.path.insert(1, '/home/hpeter/Documents/ResearchBoos/EBM_files/EBM/MEBM')

from MEBM import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=513, max_iters=1e3, tol=1e-5, diffusivity="constant")

# model.set_init_temp(init_temp_type="legendre", low=250, high=300)
model.set_init_temp(init_temp_type="load_data", low=250, high=300)

model.set_insol(insol_type="perturbation", perturb_center=15, perturb_spread=4.94, perturb_intensity=15)
# model.set_insol(insol_type="perturbation", perturb_center=60, perturb_spread=9.89, perturb_intensity=15)
        
model.set_albedo(al_feedback=False, alb_ice=0.6, alb_water=0.2)
# model.set_albedo(al_feedback=True, alb_ice=0.6, alb_water=0.2)

# model.set_olr(olr_type="planck", A=None, B=None, emissivity=0.65)
model.set_olr(olr_type="full_radiation", A=None, B=None, emissivity=None)

model.solve(numerical_method="multigrid", frames=100)

model.save_data(control=False)

model.log_efe(fname_efe="itcz.log")

model.log_feedbacks(fname_feedbacks="feedbacks.log")

model.save_plots()
