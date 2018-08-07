#!/usr/bin/env python

################################################################################
### IMPORTS
################################################################################
import numpy as np
from scipy.integrate import quadrature, trapz
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import os

################################################################################
### CONSTANTS 
################################################################################
ps  = 98000    #Pa = kg/m/s2
cp  = 1005     #J/kg/K
g   = 9.81     #m/s2
D   = 1.06e6   #m2/s
Re  = 6.371e6  #m
RH  = 0.8      #0-1
S0  = 1365     #J/m2/s
R   = 287.058  #J/kg/K
Lv  = 2257000  #J/kg
sig = 5.67e-8  #J/s/m2/K4

################################################################################
### STYLES
################################################################################
rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, 
        linewidth=1.5)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=5, width=1.5)
rc('ytick', labelsize=13)
rc('ytick.major', size=5, width=1.5)
rc('legend', fontsize=14)

EBM_PATH = os.environ['EBM_PATH']

################################################################################
### FLUXES
################################################################################
print('\nPlotting Fluxes')

T_f = np.load('T_array.npz')['arr_0'][-1, :]
E_f = np.load('E_array.npz')['arr_0'][-1, :]

dlat = 180 / (T_f.shape[0] - 1)
dlat_rad = np.deg2rad(dlat)
lats = np.linspace(-90 + dlat/2, 90 - dlat/2, T_f.shape[0])
cos_lats = np.cos(np.deg2rad(lats))
sin_lats = np.sin(np.deg2rad(lats))

# Total
flux_total = (2 * np.pi * Re * cos_lats) * (- ps / g * D / Re) * np.gradient(E_f, dlat_rad)

# Planck
emissivity = 0.6
L_planck = emissivity * sig * T_f**4

# L_avg = trapz( L_planck * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / (4 * np.pi * Re**2)
L_avg = trapz( L_planck * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

flux_planck = np.zeros( L_planck.shape )
for i in range(L_planck.shape[0]):
    flux_planck[i] = trapz( (L_planck[:i+1] - L_avg) * 2 * np.pi * Re**2 * cos_lats[:i+1], dx=dlat_rad)

# Water Vapor
L_f = np.load('L_array.npz')['arr_0'][-1, :]
L_f_const_q = np.load(EBM_PATH + '/data/L_array_constant_q_steps_mid_level_gaussian5_n361.npz')['arr_0'][-1, :]

L_avg = trapz( (L_f - L_f_const_q) * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

flux_wv = np.zeros( L_f.shape )
for i in range(L_f.shape[0]):
    flux_wv[i] = trapz( ( (L_f[:i+1] - L_f_const_q[:i+1]) - L_avg) * 2 * np.pi * Re**2 * cos_lats[:i+1], dx=dlat_rad)

# Plot
f, ax = plt.subplots(1, figsize=(16, 10))
ax.plot(sin_lats, 10**-15 * flux_planck, 'r', label='Planck')
ax.plot(sin_lats, 10**-15 * flux_wv, 'b', label='Water Vapor')
ax.plot(sin_lats, 10**-15 * flux_total, 'k', label='$K \\nabla E$')
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
ax.grid()
ax.legend(loc='upper left')
ax.set_title("Flux Distributions")
ax.set_xlabel("Lat")
ax.set_ylabel("PW")

plt.tight_layout()

fname = 'fluxes.png'
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()
################################################################################
