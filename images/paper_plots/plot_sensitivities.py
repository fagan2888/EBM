#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quadrature
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")


def get_data(filename, location):
    filename   = '/home/hpeter/Documents/ResearchBoos/EBM_files/EBM/data/' + filename
    data_array = np.loadtxt(filename, delimiter=',')
    if location == 'tropics':
        data_array = data_array[np.where(data_array[:, 0] == 15)]
    elif location == 'extratropics':
        data_array = data_array[np.where(data_array[:, 0] == 60)]
    centers     = data_array[:, 0]
    spreads     = data_array[:, 1]
    intensities = data_array[:, 2]
    efes        = data_array[:, 3]
    return centers, spreads, intensities, efes

# two plots: tropics and extratropics perturbations
f, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
ax1 = axes[0]; ax2 = axes[1]

# dictionary of 'file' : ['label', 'color', 'marker'] elements
files = {
        'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
        'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', '*'],
        'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'v'],
        'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 's'],
        'sensitivity_full_radiation_rh.dat': ['MEBM Parameterized RH Feedback', 'c', '^'],
        # 'sensitivity_full_radiation_no_wv_no_al.dat': ['CliMT No WV/AL Feedback', 'm'],
        # 'sensitivity_full_radiation_no_lr_no_al.dat': ['CliMT No LR/AL Feedback', 'y'],
        # 'sensitivity_planck.dat': ['Planck Radiation ($\\epsilon=0.65$)', 'red'],
        # 'sensitivity_linear.dat': ['Linear Radiation ($A=-572.3, B=2.92$)', 'brown'],
        # 'sensitivity_linear1.dat': ['Linear Radiation ($A=-281.7, B=1.8$)', 'pink'],
        }

# do the Clark data separately (it needs scaling)
color = 'k'
alpha = 0.5
linestyle = '--'
markersize = 8
linewidth = 1
marker = 'v'
centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'tropics')
ax1.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Prescribed WV (Clark et al.)', markersize=markersize)
centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'extratropics')
ax2.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Prescribed WV (Clark et al.)', markersize=markersize)

marker = 'o'
centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'tropics')
ax1.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Interactive WV (Clark et al.)', markersize=markersize)
centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'extratropics')
ax2.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Interactive WV (Clark et al.)', markersize=markersize)

# CESM
color = 'k'
alpha = 1.0
linestyle = '-'
markersize = 8
linewidth = 1
marker = 'o'
centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "tropics")
ax1.plot(intensities, efes, marker=marker, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label="CESM2", markersize=markersize)
centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "extratropics")
ax2.plot(intensities, efes, marker=marker, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label="CESM2", markersize=markersize)

# plot all the data
for f in files:
    for i, location in enumerate(['tropics', 'extratropics']):
        centers, spreads, intensities, efes = get_data(f, location)
        axes[i].plot(intensities, efes, marker=files[f][2], color=files[f][1], linestyle='', label=files[f][0], markersize=15)

ax1.set_xlim(0, 20)
# ax1.set_xticks([1, 3, 5, 10, 15, 18])
ax1.set_xticks([5, 10, 15, 18])
ax1.set_ylim(-16, 0)
ax1.set_yticks(np.arange(-16, 1, 2))
ax1.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
# ax1.legend(loc='lower left')
ax1.set_title('(a) Tropics', pad=10)
ax1.set_xlabel('M [W/m$^2$]')
ax1.set_ylabel('EFE Latitude')
ax1.grid()

ax2.set_xlim(0, 20)
# ax2.set_xticks([1, 3, 5, 10, 15, 18])
ax2.set_xticks([5, 10, 15, 18])
ax2.set_ylim(-16, 0)
ax2.legend(loc='lower left')
ax2.set_title('(b) Extratropics', pad=10)
ax2.set_xlabel('M [W/m$^2$]')
ax2.grid()

plt.tight_layout()

fname = 'sensitivities.png'
plt.savefig(fname, dpi=120)
plt.show()
print('{} created.'.format(fname))
plt.close()


#################################################################################
#### PERTURBATIONS
#################################################################################
#get_S_control = lambda lat: S0*np.cos(np.deg2rad(lat))/np.pi
#lat0 = 15
#sigma = 4.94
##     lat0 = 60
##     sigma = 9.89
#func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(lat0))**2 / (2*np.deg2rad(sigma)**2)) * np.cos(y)
#M0, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
#get_dS = lambda lat: - M/M0 * np.exp(-(lat - lat0)**2 / (2*sigma**2))
#get_S = lambda lat: get_S_control(lat) + get_dS(lat)
#
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5), sharey=True)
#
#ax1.set_xlabel('Latitude (degrees)')
#ax2.set_xlabel('Latitude (degrees)')
#ax1.set_ylabel('$\\overline{\\delta S}$ (W/m$^2$)')
#ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])
#ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])
#ax1.set_title('Tropics Perturbations')
#ax2.set_title('Extratropics Perturbations')
#color_cycle = ['c', 'm', 'b', 'g']
#ax1.set_prop_cycle(cycler('color', color_cycle))
#ax2.set_prop_cycle(cycler('color', color_cycle))
#
#lats = np.linspace(-90, 90, 1000)
#
#for M in [5, 10, 15, 18]:
#    sigma = 4.94; lat0 = 15
#    func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(lat0))**2 / (2*np.deg2rad(sigma)**2)) * np.cos(y)
#    M0, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
#    get_dS = lambda lat: - M/M0 * np.exp(-(lat - lat0)**2 / (2*sigma**2))
#    get_S = lambda lat: get_S_control(lat) + get_dS(lat)
#    ax1.plot(lats, get_dS(lats), label='M = {}'.format(M))
#    
#    sigma = 9.89; lat0 = 60
#    func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(lat0))**2 / (2*np.deg2rad(sigma)**2)) * np.cos(y)
#    M0, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
#    get_dS = lambda lat: - M/M0 * np.exp(-(lat - lat0)**2 / (2*sigma**2))
#    get_S = lambda lat: get_S_control(lat) + get_dS(lat)
#    ax2.plot(lats, get_dS(lats), label='M = {}'.format(M))
#
#ax1.legend()
## ax2.legend()
#
#plt.tight_layout()
#
#fname = 'gaussian_perturbs.png'
#plt.savefig(fname, dpi=120)
#print('{} created.'.format(fname))
#plt.close()
