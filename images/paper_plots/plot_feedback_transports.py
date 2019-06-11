#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

rc("font", size=10)

f, axes = plt.subplots(2, 2, figsize=(8, 2*2.47))

data = np.load("feedback_transports_differences_T15.npz")
EFE = data["EFE"] 
sin_lats = data["sin_lats"] 
delta_flux_total = data["delta_flux_total"] 
delta_flux_pl = data["delta_flux_pl"] 
delta_flux_wv = data["delta_flux_wv"] 
delta_flux_lr = data["delta_flux_lr"] 
delta_flux_al = data["delta_flux_al"] 
delta_flux_dS = data["delta_flux_dS"] 
L_pl = data["L_pl"] 
L_wv = data["L_wv"] 
L_lr = data["L_lr"] 
dS = data["dS"]
dalb = data["dalb"]
S = data["S"]

ax = axes[0, 0]
ax.plot(sin_lats, dS, "c")
ax.plot(sin_lats, dS*dalb, "c--")
ax.plot(sin_lats, -S*dalb, "g")
ax.plot(sin_lats, -L_pl, "r")
ax.plot(sin_lats, -L_wv, "m")
ax.plot(sin_lats, -L_lr, "y")
ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-150, 100])
# ax.legend(loc="upper left")
ax.set_title("(a) Tropical Differences")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Perturbation [W m$^{-2}$]")

ax = axes[0, 1]
ax.plot(sin_lats, 10**-15 * delta_flux_dS, "c")
ax.plot(sin_lats, 10**-15 * delta_flux_pl, "r")
ax.plot(sin_lats, 10**-15 * delta_flux_wv, "m")
ax.plot(sin_lats, 10**-15 * delta_flux_lr, "y")
ax.plot(sin_lats, 10**-15 * delta_flux_al, "g")
ax.plot(sin_lats, 10**-15 * delta_flux_total, "k")
ax.plot(sin_lats, 10**-15 * (delta_flux_dS + delta_flux_pl + delta_flux_wv + delta_flux_lr + delta_flux_al), "k--")
ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-5, 4])
# ax.legend(loc="upper left")
ax.set_title("(b) Tropical Transports")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")

data = np.load("feedback_transports_differences_E15.npz")
EFE = data["EFE"] 
sin_lats = data["sin_lats"] 
delta_flux_total = data["delta_flux_total"] 
delta_flux_pl = data["delta_flux_pl"] 
delta_flux_wv = data["delta_flux_wv"] 
delta_flux_lr = data["delta_flux_lr"] 
delta_flux_al = data["delta_flux_al"] 
delta_flux_dS = data["delta_flux_dS"] 
L_pl = data["L_pl"] 
L_wv = data["L_wv"] 
L_lr = data["L_lr"] 
dS = data["dS"]
dalb = data["dalb"]
S = data["S"]

ax = axes[1, 0]
l1, = ax.plot(sin_lats, dS, "c")
l2, = ax.plot(sin_lats, dS*dalb, "c--")
l3, = ax.plot(sin_lats, -S*dalb, "g")
l4, = ax.plot(sin_lats, -L_pl, "r")
l5, = ax.plot(sin_lats, -L_wv, "m")
l6, = ax.plot(sin_lats, -L_lr, "y")
l7, = ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-150, 100])
# ax.legend(loc="upper left")
ax.set_title("(a) Extratropical Differences")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Perturbation [W m$^{-2}$]")

ax = axes[1, 1]
ax.plot(sin_lats, 10**-15 * delta_flux_dS, "c")
ax.plot(sin_lats, 10**-15 * delta_flux_pl, "r")
ax.plot(sin_lats, 10**-15 * delta_flux_wv, "m")
ax.plot(sin_lats, 10**-15 * delta_flux_lr, "y")
ax.plot(sin_lats, 10**-15 * delta_flux_al, "g")
l8, = ax.plot(sin_lats, 10**-15 * delta_flux_total, "k")
l9, = ax.plot(sin_lats, 10**-15 * (delta_flux_dS + delta_flux_pl + delta_flux_wv + delta_flux_lr + delta_flux_al), "k--")
ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-5, 4])
# ax.legend(loc="upper left")
ax.set_title("(b) Extratropical Transports")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")

handles = (l1, l2, l3, l4, l5, l6, l7, l8, l9)
labels = ("$S'$", "$S'\\alpha'$", "$S\\alpha'$", "PL", "WV", "LR", "EFE", "Total", "Sum")
f.legend(handles, labels, loc="upper center", ncol=9)

plt.tight_layout()
plt.subplots_adjust(top=0.89)

fname = "differences_transports.pdf"
plt.savefig(fname)
print("{} created.".format(fname))
plt.close()
