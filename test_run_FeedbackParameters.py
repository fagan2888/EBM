#!/usr/bin/env python

from EBM_FeedbackParameters import EnergyBalanceModel

model = EnergyBalanceModel(N_pts=401, dtmax_multiple=5.0, max_sim_years=5.0, tol=1e-9)

model.insolation(insolation_type='constant', perturb_center=15, perturb_spread=4.94, perturb_intensity=18)

model.feedback_parameters(feedback_parameters_type='subtropics_only')

model.solve(numerical_method='implicit', frames=100)

model.save_data()

# model.log_efe(fname_efe=)

# model.log_feedbacks(fname_feedbacks=)

model.save_plots()
