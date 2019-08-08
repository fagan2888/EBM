import mebm 

model = mebm.MoistEnergyBalanceModel(N_pts=2**9+1, max_iters=1e3, tol=1e-4, diffusivity="constant", control_file="default")
model.set_init_temp(init_temp_type="legendre", low=250, high=300)
model.set_insol(insol_type="perturbation", perturb_center=15, perturb_spread=4.94, perturb_intensity=15)
# model.set_insol(insol_type="perturbation", perturb_center=60, perturb_spread=9.89, perturb_intensity=15)
model.set_albedo(al_feedback=True, alb_ice=0.6, alb_water=0.2)
model.set_olr(olr_type="full_radiation_rh", A=None, B=None, emissivity=None)
model.solve(numerical_method="multigrid", frames=100)
model.save_data(control=False)
model.log_efe(fname_efe="itcz.log")
model.log_feedbacks(fname_feedbacks="feedbacks.log")
model.save_plots()
