#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

EBM_PATH = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM"
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

bars_dict = {
    "C18"      : {"trop" : -4.29, "extrop" : -1.58, "color" : "white"}, 
    "CESM2"    : {"trop" : -5.16, "extrop" : -1.89, "color" : "white"}, 
    "MEBM"     : {"trop" : -3.08, "extrop" : -1.58, "color" : "white"}, 
    "MEBM Sum" : {"trop" : -3.24, "extrop" : -1.55, "color" : "gray"}, 
    "MEBM NF"  : {"trop" : -3.45, "extrop" : -3.42, "color" : "c"}, 
    "PL"       : {"trop" : -0.70, "extrop" : -4.42, "color" : "r"},   
    "WV"       : {"trop" :  0.75, "extrop" :  2.10, "color" : "m"}, 
    "AL"       : {"trop" :  0.10, "extrop" :  1.71, "color" : "g"}, 
    "LR"       : {"trop" : -0.22, "extrop" : -0.61, "color" : "y"}   
}
n_left = 5
n_right = 4

xvals = np.arange(2*len(bars_dict))
labels = []
for key in bars_dict:
    labels.append(key)

# Plot
gs = gridspec.GridSpec(1, 2, width_ratios=[n_left, n_right])

f = plt.figure(figsize=(3.404, 3.404/1.62))

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

for i, k in enumerate(labels):
    xvals[2*i+2:] += 1
    if i < n_left:
        ax = ax1
    else:
        ax = ax2
    ax.bar(xvals[2*i],   bars_dict[k]["trop"],   color=bars_dict[k]["color"], edgecolor="k", lw=0.5, label="Tropical" if i==0 else "")
    ax.bar(xvals[2*i+1], bars_dict[k]["extrop"], color=bars_dict[k]["color"], edgecolor="k", lw=0.5, hatch="////", label="Extratropical" if i==0 else "")

ax1.axhline(0, color="k", lw=0.5)
ax2.axhline(0, color="k", lw=0.5)

ax1.set_ylabel("Total Feedback, $\\lambda$ (degrees PW$^{-1}$)")
ax2.set_ylabel("Feedback Factor, $f_{i}$")
ax1.set_ylim([-5.5, 2.5])
ax2.set_ylim([-5.5, 2.5])

ax1.set_xticks(xvals[::2]+0.5)
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_xlim([xvals[0]-1.0, xvals[2*n_left-1]+1.0])

ax2.set_xticks(xvals[::2]+0.5)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_xlim([xvals[2*n_left]-1.0, xvals[-1]+1.0])

ax1.legend()

plt.tight_layout()

fname = 'feedback_factors.pdf'
# fname = 'feedback_factors.png'
plt.savefig(fname)
print('{} created.'.format(fname))
plt.close()
