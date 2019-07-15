#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from plot_sensitivities import get_data
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

D   = 1.06e6
ps  = 98000    
g   = 9.81     
Re  = 6.371e6  

cesm2_data = np.load(EBM_PATH + "/data/D_cesm2.npz")
sin_lats_cesm2 = cesm2_data["sin_lats"]
D_cesm2 = cesm2_data["D"]

N_pts = 401
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)
lats = np.arcsin(sin_lats)

def D_f(lats):
    # D_avg = 2.54523e-4
    D_avg = ps / g * D / Re**2
    lat0 = 15
    L_trop = 2 * np.sin(np.deg2rad(lat0))
    L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
    D_trop = 4.5e-4
    D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
    Diff =  D_trop * np.ones(lats.shape)
    Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
    return g/ps*Re**2 * Diff
D1 = D_f(lats)

def D_f(lats):
    # D_avg = 2.54523e-4
    D_avg = ps / g * D / Re**2
    lat0 = 15
    L_trop = 2 * np.sin(np.deg2rad(lat0))
    L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
    D_trop = 0.5e-4
    D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
    Diff =  D_trop * np.ones(lats.shape)
    Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
    return g/ps*Re**2 * Diff
D2 = D_f(lats)

f = plt.figure(figsize=(7.057, 7.057/1.62/2))

outer = gridspec.GridSpec(1, 2, width_ratios=[3, 0.84])
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.4, width_ratios=[2, 1])
gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], wspace=0)

ax1 = plt.subplot(gs1[0])
colors = [(1.0, 0.5, 0.3), (0.2, 0.8, 0.2), (0.0, 0.5, 0.8), (0.0, 0.0, 0.0)]
l1, = ax1.plot(sin_lats_cesm2, D_cesm2, c=colors[0], ls="-", label="CESM2")
l2, = ax1.plot(sin_lats, ps/g*D1/Re**2, c=colors[1], ls="--", label="$D_1$")
l3, = ax1.plot(sin_lats, ps/g*D2/Re**2, c=colors[2], ls="-.", label="$D_2$")
l4, = ax1.plot([-1, 1],[2.6e-4,2.6e-4], c=colors[3], ls=":", label="Constant")

ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax1.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax1.set_ylim([0, 5e-4])
# ax1.legend(loc="upper center", ncol=4)
ax1.annotate("(a)", (0.015, 0.94), xycoords="axes fraction")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Diffusivity, $D$ (kg m$^{-2}$ s$^{-1}$)")
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax2 = plt.subplot(gs1[1])
location = "tropics"
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation.dat", location)
l8, = ax2.plot(intensities, efes, marker="o", color=colors[3], linestyle='', label="Constant $D$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D1.dat", location)
l6, = ax2.plot(intensities, efes, marker="H", color=colors[1], linestyle='', label="$D_1$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D2.dat", location)
l7, = ax2.plot(intensities, efes, marker="D", color=colors[2], linestyle='', label="$D_2$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D_cesm2.dat", location)
l5, = ax2.plot(intensities, efes, marker="P", color=colors[0], linestyle='', label="CESM $D$")

ax2.set_xlim(0, 20)
ax2.set_xticks([0, 5, 10, 15, 18])
ax2.set_ylim(-16, 0)
ax2.set_yticks(np.arange(-16, 1, 2))
ax2.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
ax2.annotate("(b)", (0.015, 0.94), xycoords="axes fraction")
ax2.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
ax2.set_ylabel('EFE Latitude')

ax3 = plt.subplot(gs2[0])
location = "extratropics"
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation.dat", location)
ax3.plot(intensities, efes, marker="o", color=colors[3], linestyle='', label="Constant $D$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D1.dat", location)
ax3.plot(intensities, efes, marker="H", color=colors[1], linestyle='', label="$D_1$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D2.dat", location)
ax3.plot(intensities, efes, marker="D", color=colors[2], linestyle='', label="$D_2$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D_cesm2.dat", location)
ax3.plot(intensities, efes, marker="P", color=colors[0], linestyle='', label="CESM $D$")

ax3.set_xlim(0, 20)
ax3.set_xticks([0, 5, 10, 15, 18])
ax3.set_ylim(-16, 0)
ax3.set_yticks(np.arange(-16, 1, 2))
ax3.set_yticklabels(['', '', '', '', '', '', '', '', ''])
ax3.annotate("(c)", (0.015, 0.94), xycoords="axes fraction")
ax3.set_xlabel('Forcing Strength, $M$ (W m$^{-2}$)')
ax3.legend(loc="lower left")

# handles = (l1, l2, l3, l4, l5, l6, l7, l8)
# labels = ("", "", "", "", "CESM2 $D$", "$D_1$", "$D_2$", "Constant $D$")
# f.legend(handles, labels, loc="right", ncol=2, columnspacing=-1)

plt.tight_layout()
# plt.subplots_adjust(right=0.85)

fname = "diffusivity_sensitivities.pdf"
plt.savefig(fname)
# plt.show()

print("{} saved.".format(fname))
