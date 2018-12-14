#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quadrature
from matplotlib import animation, rc
from matplotlib import colors as mcolors

rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=8)
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
# lat_itcz = scaling * lat_efe
scaling = 0.64 

# two plots: tropics and extratropics perturbations
f, axes = plt.subplots(1, 2, figsize=(10,6), sharey=True)
ax1 = axes[0]; ax2 = axes[1]

# dictionary of 'file' : ['label', 'color'] elements
# (see matplotlib.colors.CSS4_COLORS in a terminal to see all the names)
files = {
        # 'perturbed_efe_clark_no_wvf.dat':                    ['Prescirbed WV (Clark et al.)',              'darkorange'],
        # 'perturbed_efe_clark_wvf.dat':                       ['Interactive WV (Clark et al.)',             'darkorange'],

        # 'perturbed_efe_planck.dat':                          ['Planck OLR',                                'gray'],
        # 'perturbed_efe_planck_linear.dat':                   ['Linear Planck OLR (Solden & Held)',         'firebrick'],
        # 'perturbed_efe_planck_linear_fit.dat':               ['Linear Planck OLR Fit',                     'lawngreen'],

        'perturbed_efe_full_wvf_mid_level_gaussian5.dat':    ['CliMT Interactive WV', 'red'],
        'perturbed_efe_full_no_wvf_mid_level_gaussian5.dat': ['CliMT Prescribed WV',  'blue'],

        # 'perturbed_efe_full_wvf.dat':                        ['CliMT Interactive WV',                      'red'],
        # 'perturbed_efe_full_wvf_linear.dat':                 ['Linear Interactive WV (Solden & Held)',     'pink'],
        # 'perturbed_efe_full_wvf_linear_fit.dat':             ['Linear CliMT Interactive WV Fit',           'lightskyblue'],

        # 'perturbed_efe_full_no_wvf.dat':                     ['CliMT Prescribed WV',                       'green'],
        # 'perturbed_efe_full_no_wvf_linear.dat':              ['Linear Prescribed WV (Solden & Held)',      'black'],
        # 'perturbed_efe_full_no_wvf_linear_fit.dat':          ['Linear CliMT Prescribed WV Fit',            'purple']

        'perturbed_efe_full_wvf_alb_feedback.dat': ['CliMT Interactive WV, Albedo Feedback 0.3',  'pink'],
        'perturbed_efe_full_wvf_alb_feedback0.5.dat': ['CliMT Interactive WV, Albedo Feedback 0.5',  'cyan'],
        }
color_dict = mcolors.CSS4_COLORS

# do the Clark data separately (it doesn't need scaling and also uses markeredgewidth
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_no_wvf.dat', 'tropics')
ax1.plot(intensities, itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Prescribed WV (Clark et al.)', markerfacecolor='w', markeredgewidth=1.5)
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_wvf.dat', 'tropics')
ax1.plot(intensities, itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Interactive WV (Clark et al.)')

centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_no_wvf.dat', 'extratropics')
ax2.plot(intensities, itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Prescribed WV (Clark et al.)', markerfacecolor='w', markeredgewidth=1.5)
centers, spreads, intensities, itczs = get_data('perturbed_efe_clark_wvf.dat', 'extratropics')
ax2.plot(intensities, itczs, color=color_dict['darkorange'], marker='o', linestyle='', label='Interactive WV (Clark et al.)')

# plot all the data
for f in files:
    for i, location in enumerate(['tropics', 'extratropics']):
        centers, spreads, intensities, EFEs = get_data(f, location)
        axes[i].plot(intensities, scaling * EFEs, marker='o', color=color_dict[files[f][1]], linestyle='', label=files[f][0])

ax1.set_xlim(3, 20)
ax1.set_xticks([5, 10, 15, 18])
ax1.set_ylim(-10, 0)
ax1.set_yticklabels(['10$^\\circ$S', '8$^\\circ$S', '6$^\\circ$S', '4$^\\circ$S', '2$^\\circ$S', 'EQ'])
ax1.set_title('Tropics')
ax1.set_xlabel('M (W/m$^2$)')
ax1.set_ylabel('ITCZ Location')

ax2.set_xlim(3, 20)
ax2.set_xticks([5, 10, 15, 18])
ax2.legend(loc='lower right')
ax2.set_title('Extratropics')
ax2.set_xlabel('M (W/m$^2$)')

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
