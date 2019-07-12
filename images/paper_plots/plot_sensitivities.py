#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.integrate import quadrature
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

rc("lines", markersize=8)

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
    f, axes = plt.subplots(1, 2, figsize=(7.057, 7.057/1.62), sharey=True)
    ax1 = axes[0]; ax2 = axes[1]
    
    # dictionary of 'file' : ['label', 'color', 'marker'] elements
    files = {
            # 'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
            # 'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', '*'],
            # 'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'v'],
            # 'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 's'],
            # 'sensitivity_full_radiation_rh.dat': ['MEBM Param. RH Feedback', 'c', '^'],
            'sensitivity_full_radiation.dat': ['MEBM', 'k', 'o'],
            'sensitivity_full_radiation_no_al.dat': ['MEBM No AL Feedback', 'g', 'o'],
            'sensitivity_full_radiation_no_wv.dat': ['MEBM No WV Feedback', 'm', 'o'],
            'sensitivity_full_radiation_no_lr.dat': ['MEBM No LR Feedback', 'y', 'o'],
            'sensitivity_full_radiation_rh.dat': ['MEBM Param. RH Feedback', 'c', 'o'],
            }
    
    # plot all the data
    for f in files:
        for i, location in enumerate(['tropics', 'extratropics']):
            centers, spreads, intensities, efes = get_data(f, location)
            axes[i].plot(intensities, efes, marker=files[f][2], color=files[f][1], linestyle='', label=files[f][0])
    # C18
    color = 'k'
    alpha = 0.5
    linestyle = '--'
    markersize = 3
    linewidth = 0.5
    # marker = 'v'
    marker = 'o'
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'tropics')
    ax1.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Prescribed WV (C18)', markersize=markersize)
    centers, spreads, intensities, efes = get_data('sensitivity_clark_no_wv.dat', 'extratropics')
    ax2.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Prescribed WV (C18)', markersize=markersize)
    
    marker = 'o'
    linestyle = '-.'
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'tropics')
    ax1.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Interactive WV (C18)', markersize=markersize)
    centers, spreads, intensities, efes = get_data('sensitivity_clark.dat', 'extratropics')
    ax2.plot(intensities, efes, color=color, marker=marker, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label='Interactive WV (C18)', markersize=markersize)
    
    # CESM
    color = 'k'
    alpha = 1.0
    linestyle = '-'
    markersize = 3
    linewidth = 0.5
    marker = 'o'
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "tropics")
    ax1.plot(intensities, efes, marker=marker, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label="CESM2", markersize=markersize)
    centers, spreads, intensities, efes = get_data("sensitivity_cesm2.dat", "extratropics")
    ax2.plot(intensities, efes, marker=marker, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label="CESM2", markersize=markersize)
    
    ax1.set_xlim(3, 20)
    ax1.set_xticks([5, 10, 15, 18])
    ax1.set_ylim(-16, 0)
    ax1.set_yticks(np.arange(-16, 1, 2))
    ax1.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
    ax1.annotate("(a)", (0.02, 0.96), xycoords="axes fraction")
    ax1.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
    ax1.set_ylabel('EFE Latitude')
    
    ax2.set_xlim(3, 20)
    ax2.set_xticks([5, 10, 15, 18])
    ax2.set_ylim(-16, 0)
    ax2.legend(loc='lower left')
    ax2.annotate("(b)", (0.02, 0.96), xycoords="axes fraction")
    ax2.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
    
    plt.tight_layout()
    
    fname = 'sensitivities.pdf'
    plt.savefig(fname)
    print('{} created.'.format(fname))
    plt.close()
