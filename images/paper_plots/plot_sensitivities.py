#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quadrature
from matplotlib import rc
from matplotlib.lines import Line2D
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

# rc("lines", markersize=8)
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

if __name__ == "__main__":
    # two plots: tropics and extratropics perturbations
    f, axes = plt.subplots(1, 2, figsize=(7.057, 7.057/1.62/1.8), sharey=True)
    ax1 = axes[0]; ax2 = axes[1]
    
    # dictionary of 'file' : ['label', 'color', 'marker'] elements
    files = {
            'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
            'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', '*'],
            'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'v'],
            'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 's'],
            'sensitivity_full_radiation_rh.dat': ['MEBM RH Feedback', 'b', '^'],
            }
    
    # plot all the data
    for f in files:
        for i, location in enumerate(['tropics', 'extratropics']):
            centers, spreads, intensities, efes = get_data(f, location)
            axes[i].plot(intensities, efes, marker=files[f][2], color=files[f][1], linestyle='', label=files[f][0])
    
    # CESM
    color = 'k'
    linestyle = '-'
    marker = 'o'
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "tropics")
    ax1.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "extratropics")
    ax2.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)

    # C18
    color = 'k'
    linestyle = '--'
    marker = 'v'
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'tropics')
    ax1.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'extratropics')
    ax2.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    
    color = 'k'
    marker = 'o'
    linestyle = '--'
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'tropics')
    ax1.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'extratropics')
    ax2.plot(intensities, efes, color=color, marker=marker, markeredgecolor='k', markerfacecolor='none', linestyle=linestyle, alpha=0.5)
    
    ax1.set_xlim(0, 20)
    ax1.set_xticks([0, 5, 10, 15, 18])
    ax1.set_ylim(-16, 0)
    ax1.set_yticks(np.arange(-16, 1, 2))
    ax1.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
    ax1.annotate("(a)", (0.02, 0.93), xycoords="axes fraction")
    ax1.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
    ax1.set_ylabel('EFE Latitude, $\phi_E$')
    
    ax2.set_xlim(0, 20)
    ax2.set_xticks([0, 5, 10, 15, 18])
    legend_elements = [Line2D([0], [0], color="k", linestyle="",   marker="o", label="Control"),
                       Line2D([0], [0], color="g", linestyle="",   marker="*", label="No AL feedback"),
                       Line2D([0], [0], color="m", linestyle="",   marker="v", label="No WV feedback"),
                       Line2D([0], [0], color="y", linestyle="",   marker="s", label="No LR feedback"),
                       Line2D([0], [0], color="b", linestyle="",   marker="^", label="RH parameterization"),
                       Line2D([0], [0], color="k", linestyle="-",  alpha=0.5, marker="", label="CESM2"),
                       Line2D([0], [0], color="k", linestyle="--", alpha=0.5, marker="", label="Clark et al. (C18)")]
    ax2.legend(handles=legend_elements, loc="lower left", fontsize=6)
    ax1.annotate("open for GCM", xy=(9.8, -7.8), xycoords="data", xytext=(-60, -40), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    ax2.annotate("closed for MEBM", xy=(17.8, -7), xycoords="data", xytext=(-60, -40), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    ax2.annotate("(b)", (0.02, 0.93), xycoords="axes fraction")
    ax2.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
    
    plt.tight_layout()
    
    fname = 'sensitivities.pdf'
    plt.savefig(fname)
    print('{} created.'.format(fname))
    plt.close()
