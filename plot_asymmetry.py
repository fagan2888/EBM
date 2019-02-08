#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('lines', linewidth=4, markersize=8)
rc('axes', titlesize=20, labelsize=16, xmargin=0.02, ymargin=0.02, linewidth=2)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=8, width=2)
rc('ytick', labelsize=13)
rc('ytick.major', size=8, width=2)
rc('legend', fontsize=13)


def asym(A):
    ''' returns pointwise asymmetry in A assuming A is a 1D array '''
    L = A.shape[0]
    return A[L//2:] - np.flip(A[:L//2 + L%2], axis=0)


################################################################################
# Just plot full L and q
################################################################################
# Pick Sims to Compare
forcing_type = "Tropical"
sim_dir_pairs = [ 
                  # ['sim221','sim229', 'M=5'],
                  # ['sim222','sim230', 'M=10'],
                  ['sim223','sim231', 'M=15'],
                  # ['sim224','sim232', 'M=18'],
                ]

# forcing_type = "ExtraTropical"
# sim_dir_pairs = [ 
#                   # ['sim225','sim233', 'M=5'],
#                   # ['sim226','sim234', 'M=10'],
#                   ['sim227','sim235', 'M=15'],
#                   # ['sim228','sim236', 'M=18'],
#                 ]

# Plot setup
f = plt.figure(figsize=(10,12))

ax1 = plt.subplot(211)
ax1.set_title(forcing_type + ' Forcing: $\\overline{L}$')
ax1.set_ylabel('OLR [W m$^{-2}$]')
ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax1.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
# ax1.set_ylim([-35, 35])

# ax1.plot([0, 1], [0, 0], 'k--')

ax2 = plt.subplot(212)
ax2.set_title(forcing_type + ' Forcing: Column integrated WV')
ax2.set_ylabel('WV [kg m$^{-2}$]')
ax2.set_xlabel('Latitude')
ax2.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax2.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
# ax2.set_ylim([-70, 70])

# ax2.plot([0, 1], [0, 0], 'k--')
pressures = np.load('data/moist_adiabat_data.npz')['pressures']
for sim_dir_pair in sim_dir_pairs:
    # Load data
    L_wvf     = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[0]))['arr_0']
    q_wvf     = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[0]))['arr_0']
    L_no_wvf  = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[1]))['arr_0']
    q_no_wvf  = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[1]))['arr_0']
    
    # Use equilibrated data
    L_wvf = L_wvf[-1, :]
    L_no_wvf = L_no_wvf[-1, :]
    q_wvf = q_wvf[-1, :, :]
    q_no_wvf = q_no_wvf[-1, :, :]

    N_pts = L_wvf.shape[0]
    dx = 2 / N_pts
    sin_lats = np.linspace(-1 + dx/2, 1 - dx/2, N_pts)
    lats = np.rad2deg(np.arcsin(sin_lats))
    ################################################################################
    # plt.figure()
    # plt.plot(sin_lats, q_no_wvf[:, 0])
    # plt.show()
    ################################################################################
    
    # Vertically integrate q
    g = 9.81
    dp = pressures[0] - pressures[1]
    q_wvf_column = np.sum(q_wvf, axis=1) * dp / g 
    q_no_wvf_column = np.sum(q_no_wvf, axis=1) * dp / g
    
    ax1.plot(sin_lats, L_wvf, label=sim_dir_pair[2] + ": Interactive")
    ax1.plot(sin_lats, L_no_wvf, label=sim_dir_pair[2] + ": Prescribed")
    ax2.plot(sin_lats, q_wvf_column,  label=sim_dir_pair[2] + ": Interactive")
    ax2.plot(sin_lats, q_no_wvf_column,  label=sim_dir_pair[2] + ": Interactive")
    
ax1.legend()
ax2.legend()
plt.tight_layout()

fname = 'L_and_q.png'
plt.savefig(fname, dpi=80)
print(fname, "created")



################################################################################
# Plot diffs for L and q
################################################################################
# Pick Sims to Compare
forcing_type = "Tropical"
sim_dir_pairs = [ 
                  # ['sim221','sim229', 'M=5'],
                  # ['sim222','sim230', 'M=10'],
                  ['sim223','sim231', 'M=15'],
                  # ['sim224','sim232', 'M=18'],
                ]

# forcing_type = "ExtraTropical"
# sim_dir_pairs = [ 
#                   # ['sim225','sim233', 'M=5'],
#                   # ['sim226','sim234', 'M=10'],
#                   ['sim227','sim235', 'M=15'],
#                   # ['sim228','sim236', 'M=18'],
#                 ]

# Plot setup
f = plt.figure(figsize=(10,12))

ax1 = plt.subplot(211)
ax1.set_title(forcing_type + ' Forcing: Diff. in $\\overline{L}$')
ax2.set_ylabel('Diff. in OLR [kg m$^{-2}$]\nInteractive - Prescribed')
ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax1.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
# ax1.set_ylim([-35, 35])

# ax1.plot([0, 1], [0, 0], 'k--')

ax2 = plt.subplot(212)
ax2.set_title(forcing_type + ' Forcing: Diff. in Column integrated WV')
ax2.set_ylabel('Diff. in WV [kg m$^{-2}$]\nInteractive - Prescribed')
ax2.set_xlabel('Latitude')
ax2.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax2.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
# ax2.set_ylim([-70, 70])

# ax2.plot([0, 1], [0, 0], 'k--')
pressures = np.load('data/moist_adiabat_data.npz')['pressures']
for sim_dir_pair in sim_dir_pairs:
    # Load data
    L_wvf     = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[0]))['arr_0']
    q_wvf     = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[0]))['arr_0']
    L_no_wvf  = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[1]))['arr_0']
    q_no_wvf  = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[1]))['arr_0']
    
    # Use equilibrated data
    L_wvf = L_wvf[-1, :]
    L_no_wvf = L_no_wvf[-1, :]
    q_wvf = q_wvf[-1, :, :]
    q_no_wvf = q_no_wvf[-1, :, :]

    N_pts = L_wvf.shape[0]
    dx = 2 / N_pts
    sin_lats = np.linspace(-1 + dx/2, 1 - dx/2, N_pts)
    lats = np.rad2deg(np.arcsin(sin_lats))
    
    # Vertically integrate q
    g = 9.81
    dp = pressures[0] - pressures[1]
    q_wvf_column = np.sum(q_wvf, axis=1) * dp / g 
    q_no_wvf_column = np.sum(q_no_wvf, axis=1) * dp / g
    
    ax1.plot(sin_lats, L_wvf - L_no_wvf, label=sim_dir_pair[2])
    ax2.plot(sin_lats, q_wvf_column - q_no_wvf_column,  label=sim_dir_pair[2])
    
ax1.legend()
ax2.legend()
plt.tight_layout()

fname = 'L_and_q_diffs.png'
plt.savefig(fname, dpi=80)
print(fname, "created")

################################################################################
# Plot Asymmetry
################################################################################
# Pick Sims to Compare
forcing_type = "Tropical"
sim_dir_pairs = [ 
                  # ['sim221','sim229', 'M=5'],
                  # ['sim222','sim230', 'M=10'],
                  ['sim223','sim231', 'M=15'],
                  # ['sim224','sim232', 'M=18'],
                ]

# forcing_type = "ExtraTropical"
# sim_dir_pairs = [ 
#                   ['sim225','sim233', 'M=5'],
#                   ['sim226','sim234', 'M=10'],
#                   ['sim227','sim235', 'M=15'],
#                   ['sim228','sim236', 'M=18'],
#                 ]

# Setup plots
f = plt.figure(figsize=(10,12))

ax1 = plt.subplot(211)
ax1.set_title(forcing_type + ' Forcing: $-\\delta P(\\overline{L})$')
ax1.set_ylabel('Diff. in Asymmetry [W m$^{-2}$]\nInteractive - Prescribed')
ax1.set_xticks(np.sin(np.deg2rad(np.arange(0, 91, 10))))
ax1.set_xticklabels(['EQ', '', '', '30', '', '', '', '', '', '90'])
ax1.set_ylim([-35, 35])

ax1.plot([0, 1], [0, 0], 'k--')

ax2 = plt.subplot(212)
ax2.set_title(forcing_type + ' Forcing: $\\delta P$(Column integrated WV)')
ax2.set_ylabel('Diff. in Asymmetry [kg m$^{-2}$]\nInteractive - Prescribed')
ax2.set_xlabel('Latitude')
ax2.set_xticks(np.sin(np.deg2rad(np.arange(0, 91, 10))))
ax2.set_xticklabels(['EQ', '', '', '30', '', '', '', '', '', '90'])
ax2.set_ylim([-70, 70])

ax2.plot([0, 1], [0, 0], 'k--')
pressures = np.load('data/moist_adiabat_data.npz')['pressures']
for sim_dir_pair in sim_dir_pairs:
    # Load data
    L_wvf     = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[0]))['arr_0']
    q_wvf     = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[0]))['arr_0']
    L_no_wvf  = np.load('../EBM_sims/{}/L_array.npz'.format(sim_dir_pair[1]))['arr_0']
    q_no_wvf  = np.load('../EBM_sims/{}/q_array.npz'.format(sim_dir_pair[1]))['arr_0']
    
    # Use equilibrated data
    L_wvf = L_wvf[-1, :]
    L_no_wvf = L_no_wvf[-1, :]
    q_wvf = q_wvf[-1, :, :]
    q_no_wvf = q_no_wvf[-1, :, :]

    N_pts = L_wvf.shape[0]
    dx = 2 / N_pts
    sin_lats = np.linspace(0, 1 - dx/2, (N_pts - 1)/2 + 1)
    lats = np.rad2deg(np.arcsin(sin_lats))
    
    # Vertically integrate q
    g = 9.81
    dp = pressures[0] - pressures[1]
    q_wvf_column = np.sum(q_wvf, axis=1) * dp / g 
    q_no_wvf_column = np.sum(q_no_wvf, axis=1) * dp / g
    
    # Compute difference in pointwise asymmetries (see Clark et al.)
    L = asym(L_wvf) - asym(L_no_wvf)
    q = asym(q_wvf_column) - asym(q_no_wvf_column)
    
    ax1.plot(sin_lats, -L, label=sim_dir_pair[2])
    ax2.plot(sin_lats, q,  label=sim_dir_pair[2])
    #ax2.plot([np.sin(np.deg2rad(6.19 * 0.64)), np.sin(np.deg2rad(6.19 * 0.64))], [-100, 100], 'k-.', label='$\\theta_{ITCZ}$ (Interactive)')
    
ax1.legend()
ax2.legend()
plt.tight_layout()

fname = 'asymmetries.png'
plt.savefig(fname, dpi=80)
print(fname, "created")
