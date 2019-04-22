#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quadrature
from matplotlib import animation, rc
from matplotlib import colors as mcolors

rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.05, ymargin=0.05, linewidth=2)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=8, width=2)
rc('ytick', labelsize=13)
rc('ytick.major', size=8, width=2)
rc('legend', fontsize=9)


def get_data(filename, location):
    filename   = 'data/' + filename
    data_array = np.loadtxt(filename, delimiter=',')
    if location == 'tropics':
        data_array = data_array[np.where(data_array[:, 0] == 15)]
    elif location == 'extratropics':
        data_array = data_array[np.where(data_array[:, 0] == 60)]
    centers     = data_array[:, 0]
    spreads     = data_array[:, 1]
    intensities = data_array[:, 2]
    EFEs        = data_array[:, 3]
    return centers, spreads, intensities, EFEs

################################################################################
### SENSITIVITY COMPARISONS
################################################################################
# Clark et al. 2018 scaling: lat_itcz = scaling * lat_efe
scaling = 0.64 

# two plots: tropics and extratropics perturbations
f, axes = plt.subplots(1, 2, figsize=(10,6), sharey=True)
ax1 = axes[0]; ax2 = axes[1]

# dictionary of 'file' : ['label', 'color'] elements
# (see matplotlib.colors.CSS4_COLORS in a terminal to see all the names)
files = {
        'sensitivity_full_radiation.dat': ['CliMT', 'black'],
        'sensitivity_full_radiation_alb.dat': ['CliMT With Albedo Feedback', 'grey'],
        # 'sensitivity_full_radiation_no_wv.dat': ['CliMT No WV Feedback', 'green'],
        # 'sensitivity_full_radiation_no_lr.dat': ['CliMT No LR Feedback', 'violet'],
        # 'sensitivity_planck.dat': ['Planck Radiation ($\\epsilon=0.65$)', 'red'],
        # 'sensitivity_linear.dat': ['Linear Radiation ($A=-572.3, B=2.92$)', 'brown'],
        # 'sensitivity_linear1.dat': ['Linear Radiation ($A=-281.7, B=1.8$)', 'pink'],
        }
color_dict = mcolors.CSS4_COLORS

# do the Clark data separately (it needs scaling and also uses markeredgewidth
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_no_wvf.dat', 'tropics')
ax1.plot(intensities, 1/scaling * itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Prescribed WV (Clark et al.)', markerfacecolor='w', markeredgewidth=1.5)
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_wvf.dat', 'tropics')
ax1.plot(intensities, 1/scaling * itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Interactive WV (Clark et al.)')

centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_no_wvf.dat', 'extratropics')
ax2.plot(intensities, 1/scaling * itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Prescribed WV (Clark et al.)', markerfacecolor='w', markeredgewidth=1.5)
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_wvf.dat', 'extratropics')
ax2.plot(intensities, 1/scaling * itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Interactive WV (Clark et al.)')

# CESM
centers, spreads, intensities, EFEs = get_data("sensitivity_cesm2.dat", "tropics")
ax1.plot(intensities, EFEs, marker='o', color=color_dict["blue"], linestyle='', label="CESM2")

# plot all the data
for f in files:
    for i, location in enumerate(['tropics', 'extratropics']):
        centers, spreads, intensities, EFEs = get_data(f, location)
        axes[i].plot(intensities, EFEs, marker='o', color=color_dict[files[f][1]], linestyle='', label=files[f][0])

ax1.set_xlim(3, 20)
ax1.set_xticks([5, 10, 15, 18])
ax1.set_ylim(-16, 0)
ax1.set_yticks(np.arange(-16, 1, 2))
ax1.set_yticklabels(['16$^\\circ$S', '14$^\\circ$S', '12$^\\circ$S', '10$^\\circ$S', '8$^\\circ$S', '6$^\\circ$S', '4$^\\circ$S', '2$^\\circ$S', 'EQ'])
ax1.set_title('Tropics')
ax1.set_xlabel('M [W/m$^2$]')
ax1.set_ylabel('EFE Latitude')

ax2.set_xlim(3, 20)
ax2.set_xticks([5, 10, 15, 18])
ax2.legend(loc='lower right')
ax2.set_title('Extratropics')
ax2.set_xlabel('M [W/m$^2$]')

plt.tight_layout()

fname = 'sensitivities.png'
plt.savefig(fname, dpi=120)
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
