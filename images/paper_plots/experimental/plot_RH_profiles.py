#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

def plot_RH(ax, sin_lats, pressures, RH):
    levels = np.arange(0, 1.05, 0.05)
    cf = ax.contourf(sin_lats, pressures, RH, cmap="BrBG", levels=levels)
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax.set_yticks(np.arange(0,1001,100))
    ax.set_yticklabels(["0", "", "200", "", "400", "", "600", "", "800", "", "1000"])
    ax.invert_yaxis()
    return cf

f, axes = plt.subplots(2, 2, figsize=(7.057, 7.057/1.62))

# top left plot
data = np.load("RH_M0_cesm2.npz")
RH = data["RH"]
sin_lats = np.sin(data["lats"])
pressures = data["pressures"]
ax = axes[0, 0]
cf = plot_RH(ax, sin_lats, pressures, RH)
ax.set_xlabel("Latitude")
ax.set_ylabel("Pressure, $p$ (hPa)")
ax.annotate("(a)", (0.02, 1.05), xycoords="axes fraction")

# bottom left plot
data = np.load("RH_M18_cesm2.npz")
RH = data["RH"]
sin_lats = np.sin(data["lats"])
pressures = data["pressures"]
ax = axes[1, 0]
cf =plot_RH(ax, sin_lats, pressures, RH)
ax.set_xlabel("Latitude")
ax.set_ylabel("Pressure, $p$ (hPa)")
ax.annotate("(c)", (0.02, 1.05), xycoords="axes fraction")

# top right plot
data = np.load("RH_M0_mebm.npz")
RH = data["RH"][:, :, 0]
sin_lats = np.sin(data["lats"])
pressures = data["pressures"]/100
ax = axes[0, 1]
cf = plot_RH(ax, sin_lats, pressures, RH)
ax.set_xlabel("Latitude")
ax.set_ylabel("Pressure, $p$ (hPa)")
ax.annotate("(b)", (0.02, 1.05), xycoords="axes fraction")

# bottom right plot
data = np.load("RH_M18_mebm.npz")
RH = data["RH"][:, :, 0]
sin_lats = np.sin(data["lats"])
pressures = data["pressures"]/100
ax = axes[1, 1]
cf = plot_RH(ax, sin_lats, pressures, RH)
ax.set_xlabel("Latitude")
ax.set_ylabel("Pressure, $p$ (hPa)")
ax.annotate("(d)", (0.02, 1.05), xycoords="axes fraction")

# colorbar at bottom
f.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.2, hspace=0.3)

cax = f.add_axes([0.3, 0.08, 0.4, 0.03])
cb = plt.colorbar(cf, ax=ax, cax=cax, orientation="horizontal")
cb.set_ticks(np.arange(0, 1.05, 0.1))
cb.set_label("Relative Humidity")

fname = "RH_profiles.pdf"
plt.savefig(fname)
print("{} saved.".format(fname))
plt.close()
