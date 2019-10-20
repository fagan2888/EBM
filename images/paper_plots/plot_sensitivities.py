#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate, scipy.interpolate
from matplotlib import rc
from matplotlib.lines import Line2D
import mebm
import os

EBM_PATH = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM"
# os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

rc("lines", markersize=6)

def get_data(filename, location):
    filename   = EBM_PATH + '/data/' + filename
    data_array = np.loadtxt(filename, delimiter=',')
    if location == 'tropics':
        data_array = data_array[np.where(data_array[:, 0] == 15)]
    elif location == 'extratropics':
        data_array = data_array[np.where(data_array[:, 0] == 60)]
    centers = data_array[:, 0]
    spreads = data_array[:, 1]
    intensities = data_array[:, 2]
    efes = data_array[:, 3]
    return centers, spreads, intensities, efes

def get_dS(perturb_intensity, location):
    if location == "tropics":
        perturb_center = 15
        perturb_spread = 4.94
    elif location == "extratropics":
        perturb_center = 60
        perturb_spread = 9.89
    func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
    perturb_normalizer, er = sp.integrate.quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
    return -perturb_intensity/perturb_normalizer * np.exp(-(np.arcsin(sin_lats) - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))

if __name__ == "__main__":
    # calculate effective forcings 
    m = mebm.MoistEnergyBalanceModel(N_pts=513)
    ctrl_alb = m.ctrl_data["alb"]
    sin_lats = m.sin_lats
    intensities_t = []
    intensities_e = []
    i_eq = len(ctrl_alb)//2
    for M in [5, 10, 15, 18]:
        dS = get_dS(M, "tropics")
        dS_alb = dS*(1 - ctrl_alb)
        dS_alb_trans = 10**-15 * m._calculate_trans(dS_alb, force_zero=True)
        intensities_t.append(dS_alb_trans[i_eq])
        dS = get_dS(M, "extratropics")
        dS_alb = dS*(1 - ctrl_alb)
        dS_alb_trans = 10**-15 * m._calculate_trans(dS_alb, force_zero=True)
        intensities_e.append(dS_alb_trans[i_eq])

    # two plots: tropics and extratropics perturbations
    f, axes = plt.subplots(1, 2, figsize=(7.057, 7.057/1.62/1.8), sharey=True)
    ax1 = axes[0]; ax2 = axes[1]
    
    # dictionary of 'file' : ['label', 'color', 'marker'] elements
    files = {
            'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
            'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', '*'],
            'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'v'],
            'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 's'],
            'sensitivity_full_radiation_rh.dat': ['MEBM RH Feedback', 'b', 'h'],
            }
    
    # plot all the data
    for f in files:
        centers, spreads, intensities, efes = get_data(f, "tropics")
        ax1.plot(intensities_t, efes, marker=files[f][2], color=files[f][1], linestyle='', label=files[f][0])
        centers, spreads, intensities, efes = get_data(f, "extratropics")
        ax2.plot(intensities_e, efes, marker=files[f][2], color=files[f][1], linestyle='', label=files[f][0])
    
    # CESM
    color = 'k'
    linestyle = '-'
    marker = 'o'
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "tropics")
    ax1.plot(intensities_t, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "extratropics")
    ax2.plot(intensities_e, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)

    # C18
    color = 'k'
    linestyle = '--'
    marker = '^'
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'tropics')
    ax1.plot(intensities_t, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'extratropics')
    ax2.plot(intensities_e, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    
    color = 'k'
    marker = 'o'
    linestyle = '--'
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'tropics')
    ax1.plot(intensities_t, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'extratropics')
    ax2.plot(intensities_e, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    
    ax1.set_xlim(0, 4)
    # ax1.set_xticks([0, 5, 10, 15, 18])
    ax1.set_ylim(-16, 0)
    ax1.set_yticks(np.arange(-16, 1, 2))
    ax1.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
    ax1.annotate("(a)", (0.02, 0.93), xycoords="axes fraction")
    ax1.set_xlabel("Forcing Strength (PW)")
    ax1.set_ylabel('EFE Latitude, $\phi_E$')
    
    ax2.set_xlim(0, 4)
    # ax2.set_xticks([0, 5, 10, 15, 18])
    legend_elements = [Line2D([0], [0], color="k", linestyle="",   marker="o", label="MEBM control"),
                       Line2D([0], [0], color="g", linestyle="",   marker="*", label="MEBM no AL feedback"),
                       Line2D([0], [0], color="m", linestyle="",   marker="v", label="MEBM no WV feedback"),
                       Line2D([0], [0], color="y", linestyle="",   marker="s", label="MEBM no LR feedback"),
                       Line2D([0], [0], color="b", linestyle="",   marker="h", label="MEBM RH parameterization"),
                       Line2D([0], [0], color="k", linestyle="-",  alpha=0.5, marker="o", markerfacecolor='none', label="CESM2 control"),
                       Line2D([0], [0], color="k", linestyle="--", alpha=0.5, marker="o", markerfacecolor='none', label="C18 control"),
                       Line2D([0], [0], color="k", linestyle="--", alpha=0.5, marker="^", markerfacecolor='none', label="C18 no WV feedback")]
    ax2.legend(handles=legend_elements, loc="lower left", fontsize=6, ncol=2)
    # ax1.annotate("open for GCM", xy=(9.8, -7.8), xycoords="data", xytext=(-60, -40), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    # ax2.annotate("closed for MEBM", xy=(17.8, -7), xycoords="data", xytext=(-60, -40), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    ax2.annotate("(b)", (0.02, 0.93), xycoords="axes fraction")
    ax2.set_xlabel("Forcing Strength (PW)")
    
    plt.tight_layout()
    
    fname = 'sensitivities.pdf'
    plt.savefig(fname)
    print('{} created.'.format(fname))
    plt.close()
