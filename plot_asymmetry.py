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
    return A[L//2:] - np.flip(A[:L//2], axis=0)

# Load data
L_wvf = np.load('data/L_array_full_wvf_perturbation.npz')['arr_0']
L_no_wvf = np.load('data/L_array_full_no_wvf_perturbation.npz')['arr_0']
q_wvf = np.load('data/q_array_full_wvf_perturbation.npz')['arr_0']
q_no_wvf = np.load('data/q_array_full_no_wvf_perturbation.npz')['arr_0']
pressures = np.load('data/moist_adiabat_data.npz')['pressures']

# Use equilibrated data
L_wvf = L_wvf[-1, :]
L_no_wvf = L_no_wvf[-1, :]
i = -1
while q_wvf[i, 0, 0] == 0:
    i -= 1
q_wvf = q_wvf[i, :, :]
# print(q_wvf[180,:])
# print(pressures)
i = -1
while q_no_wvf[i, 0, 0] == 0:
    i -= 1
q_no_wvf = q_no_wvf[i, :, :]

# Vertically integrate q
g = 9.81
dp = pressures[0] - pressures[1]
q_wvf_column = np.sum(q_wvf, axis=1) * dp / g 
q_no_wvf_column = np.sum(q_no_wvf, axis=1) * dp / g

# Compute difference in pointwise asymmetries (see Clark et al.)
L = asym(L_wvf) - asym(L_no_wvf)
q = asym(q_wvf_column) - asym(q_no_wvf_column)

# Plot
lats = np.linspace(0, 90, L.shape[0])
f = plt.figure(figsize=(10,12))

ax = plt.subplot(211)
ax.set_title('Tropical Forcing')
ax.set_ylabel('Diif. in Asymmetry [W m$^{-2}$]\nInteractive - Prescribed')
ax.set_xticks([0, 30, 60, 90])
ax.set_ylim([-np.max(np.abs(L)) - 1, np.max(np.abs(L)) + 1])

ax.plot(lats, -L, label='$-\\delta P(\\bar L)$')
ax.plot([0, 90], [0, 0], 'k--')
ax.legend()

ax = plt.subplot(212)
ax.set_title('Tropical Forcing')
ax.set_ylabel('Diif. in Asymmetry [kg m$^{-2}$]\nInteractive - Prescribed')
ax.set_xlabel('Latitude')
ax.set_xticks([0, 30, 60, 90])
ax.set_ylim([np.min(q) - 1, -np.min(q) + 1])

ax.plot(lats, q, label='$\\delta P($Column Integrated WV)')
ax.plot([0, 90], [0, 0], 'k--')
ax.plot([6.19 * 0.64, 6.19 * 0.64], [np.min(q), np.max(q)], 'k-.', label='$\\theta_{ITCZ}$ (Interactive)')
ax.legend()

plt.tight_layout()


plt.show()
