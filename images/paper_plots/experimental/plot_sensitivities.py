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
    f, ax = plt.subplots(1)
    
    # dictionary of 'file' : ['label', 'color', 'marker'] elements
    files = {
            'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
            'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', '*'],
            'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'v'],
            'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 's'],
            'sensitivity_full_radiation_rh.dat': ['MEBM RH Feedback', 'c', '^'],
            }
    
    # plot all the data
    for f in files:
        for i, location in enumerate(['tropics', 'extratropics']):
            centers, spreads, intensities, efes = get_data(f, location)
            if location=='tropics':
                ax.plot(intensities, efes, marker=files[f][2], color='r', linestyle='-', label=files[f][0], alpha=0.5)
            else:
                ax.plot(intensities, efes, marker=files[f][2], color='b', linestyle='--', alpha=0.5)
    
    # CESM
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "tropics")
    ax.plot(intensities, efes, marker="o", color="k", alpha=0.5, linestyle="-", label="CESM2")
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "extratropics")
    ax.plot(intensities, efes, marker="o", color="k", alpha=0.5, linestyle="--", label="CESM2")

    # # C18
    # color = 'k'
    # alpha = 0.5
    # linestyle = '--'
    # markersize = 4
    # linewidth = 0.5
    # marker = 'v'
    # centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'tropics')
    # ax.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Prescribed WV (C18)', markersize=markersize)
    # centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'extratropics')
    # ax.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, markersize=markersize)
    
    # marker = 'o'
    # linestyle = '-.'
    # centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'tropics')
    # ax.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Interactive WV (C18)', markersize=markersize)
    # centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'extratropics')
    # ax.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, markersize=markersize)
    
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 18])
    ax.set_ylim(-16, 1)
    ax.set_yticks(np.arange(-16, 1, 2))
    ax.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
    ax.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
    ax.set_ylabel('EFE Latitude, $\phi_E$')
    legend_elements = [Line2D([0], [0], color="k", linestyle="", marker="o", alpha=0.5, label="Control"),
                       Line2D([0], [0], color="k", linestyle="", marker="*", alpha=0.5, label="No AL"),
                       Line2D([0], [0], color="k", linestyle="", marker="v", alpha=0.5, label="No WV"),
                       Line2D([0], [0], color="k", linestyle="", marker="s", alpha=0.5, label="No LR"),
                       Line2D([0], [0], color="k", linestyle="", marker="^", alpha=0.5, label="RH"),
                       Line2D([0], [0], color="r", linestyle="-", marker="", alpha=0.5, label="Tropical"),
                       Line2D([0], [0], color="b", linestyle="--", marker="", alpha=0.5, label="Extratropical")]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)
    
    plt.tight_layout()
    
    fname = 'sensitivities.pdf'
    plt.savefig(fname)
    print('{} created.'.format(fname))
    plt.close()
