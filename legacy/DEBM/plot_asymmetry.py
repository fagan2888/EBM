#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('lines', linewidth=3, color='b', markersize=8)
rc('axes', titlesize=20, labelsize=16, xmargin=0, ymargin=0, linewidth=2)
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

lats = np.linspace(0, 90, 181)

f = plt.figure(figsize=(10,12))

ax1 = plt.subplot(211)
ax1.set_title('Tropical Forcing')
ax1.set_ylabel('Diif. in Asymmetry [W m$^{-2}$]\nInteractive - Prescribed')
ax1.set_xticks([0, 0.5, np.sqrt(3)/2, 1])
ax1.set_xticklabels(['EQ', '30', '60', '90'])
ax1.set_ylim([-25, 25])

ax1.plot([0, 1], [0, 0], 'k--')

ax2 = plt.subplot(212)
ax2.set_title('Tropical Forcing')
ax2.set_ylabel('Diif. in Asymmetry [kg m$^{-2}$]\nInteractive - Prescribed')
ax2.set_xlabel('Latitude')
ax2.set_xticks([0, 0.5, np.sqrt(3)/2, 1])
ax2.set_xticklabels(['EQ', '30', '60', '90'])
ax2.set_ylim([-50, 50])

ax2.plot([0, 1], [0, 0], 'k--')

sim_dir_pairs = [ ['sim24','sim25'],
                  ['sim26','sim27'],
                  ['sim28','sim29'],
                  ['sim30','sim31'],
                  ['sim32','sim33']
                  ]

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
    
    # Vertically integrate q
    g = 9.81
    dp = pressures[0] - pressures[1]
    q_wvf_column = np.sum(q_wvf, axis=1) * dp / g 
    q_no_wvf_column = np.sum(q_no_wvf, axis=1) * dp / g
    
    # Compute difference in pointwise asymmetries (see Clark et al.)
    L = asym(L_wvf) - asym(L_no_wvf)
    q = asym(q_wvf_column) - asym(q_no_wvf_column)
    
    ax1.plot(np.sin(np.deg2rad(lats)), -L, label=sim_dir_pair)
    ax2.plot(np.sin(np.deg2rad(lats)), q, label=sim_dir_pair)
    #ax2.plot([np.sin(np.deg2rad(6.19 * 0.64)), np.sin(np.deg2rad(6.19 * 0.64))], [-100, 100], 'k-.', label='$\\theta_{ITCZ}$ (Interactive)')
    
ax1.legend()
ax2.legend()
plt.tight_layout()

plt.savefig('asymmetries.png')
