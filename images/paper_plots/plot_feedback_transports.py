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
dtrans_total = data["dtrans_total"] 
dtrans_pl = data["dtrans_pl"] 
dtrans_wv = data["dtrans_wv"] 
dtrans_lr = data["dtrans_lr"] 
dtrans_dS = data["dtrans_dS"] 
dtrans_S_dalb = data["dtrans_S_dalb"] 
dtrans_dS_alb = data["dtrans_dS_alb"] 
dtrans_dS_dalb = data["dtrans_dS_dalb"] 
dL_pl = data["dL_pl"] 
dL_wv = data["dL_wv"] 
dL_lr = data["dL_lr"] 
dL = data["dL"] 
dS = data["dS"]
dalb = data["dalb"]
alb = data["alb"]
S = data["S"]

ax = axes[0, 0]
l1, = ax.plot(sin_lats, dS, "c")
l2, = ax.plot(sin_lats, -S*dalb, "g")
l3, = ax.plot(sin_lats, -dS*alb, "g--")
l4, = ax.plot(sin_lats, -dS*dalb, "g-.")
l5, = ax.plot(sin_lats, dL_pl, "r")
l6, = ax.plot(sin_lats, dL_wv, "m")
l7, = ax.plot(sin_lats, dL_lr, "y")
l8, = ax.plot(sin_lats, dS - S*dalb - dS*alb - dS*dalb - dL, "k")
l9, = ax.plot(sin_lats, dS - S*dalb - dS*alb - dS*dalb + dL_pl + dL_wv + dL_lr, "k--")
l10, = ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_yticks(np.arange(-150, 101, 50))
ax.set_ylim([-155, 105])
ax.set_title("(a) Tropical Differences")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Perturbation [W m$^{-2}$]")

ax = axes[0, 1]
ax.plot(sin_lats, 10**-15 * dtrans_dS, "c")
ax.plot(sin_lats, 10**-15 * dtrans_S_dalb, "g")
ax.plot(sin_lats, 10**-15 * dtrans_dS_alb, "g--")
ax.plot(sin_lats, 10**-15 * dtrans_dS_dalb, "g-.")
ax.plot(sin_lats, 10**-15 * dtrans_pl, "r")
ax.plot(sin_lats, 10**-15 * dtrans_wv, "m")
ax.plot(sin_lats, 10**-15 * dtrans_lr, "y")
ax.plot(sin_lats, 10**-15 * dtrans_total, "k")
ax.plot(sin_lats, 10**-15 * (dtrans_dS + dtrans_S_dalb + dtrans_dS_alb + dtrans_dS_dalb + dtrans_pl + dtrans_wv + dtrans_lr), "k--")
ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-5.5, 6.5])
ax.set_title("(b) Tropical Transports")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")

################################################################################

data = np.load("feedback_transports_differences_E15.npz")
EFE = data["EFE"] 
sin_lats = data["sin_lats"] 
dtrans_total = data["dtrans_total"] 
dtrans_pl = data["dtrans_pl"] 
dtrans_wv = data["dtrans_wv"] 
dtrans_lr = data["dtrans_lr"] 
dtrans_dS = data["dtrans_dS"] 
dtrans_S_dalb = data["dtrans_S_dalb"] 
dtrans_dS_alb = data["dtrans_dS_alb"] 
dtrans_dS_dalb = data["dtrans_dS_dalb"] 
dL_pl = data["dL_pl"] 
dL_wv = data["dL_wv"] 
dL_lr = data["dL_lr"] 
dL = data["dL"] 
dS = data["dS"]
dalb = data["dalb"]
alb = data["alb"]
S = data["S"]

ax = axes[1, 0]
l1, = ax.plot(sin_lats, dS, "c")
l2, = ax.plot(sin_lats, -S*dalb, "g")
l3, = ax.plot(sin_lats, -dS*alb, "g--")
l4, = ax.plot(sin_lats, -dS*dalb, "g-.")
l5, = ax.plot(sin_lats, dL_pl, "r")
l6, = ax.plot(sin_lats, dL_wv, "m")
l7, = ax.plot(sin_lats, dL_lr, "y")
l8, = ax.plot(sin_lats, dS - S*dalb - dS*alb - dS*dalb - dL, "k")
l9, = ax.plot(sin_lats, dS - S*dalb - dS*alb - dS*dalb + dL_pl + dL_wv + dL_lr, "k--")
l10, = ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_yticks(np.arange(-150, 101, 50))
ax.set_ylim([-155, 105])
ax.set_title("(c) Extratropical Differences")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Perturbation [W m$^{-2}$]")

ax = axes[1, 1]
ax.plot(sin_lats, 10**-15 * dtrans_dS, "c")
ax.plot(sin_lats, 10**-15 * dtrans_S_dalb, "g")
ax.plot(sin_lats, 10**-15 * dtrans_dS_alb, "g--")
ax.plot(sin_lats, 10**-15 * dtrans_dS_dalb, "g-.")
ax.plot(sin_lats, 10**-15 * dtrans_pl, "r")
ax.plot(sin_lats, 10**-15 * dtrans_wv, "m")
ax.plot(sin_lats, 10**-15 * dtrans_lr, "y")
ax.plot(sin_lats, 10**-15 * dtrans_total, "k")
ax.plot(sin_lats, 10**-15 * (dtrans_dS + dtrans_S_dalb + dtrans_dS_alb + dtrans_dS_dalb + dtrans_pl + dtrans_wv + dtrans_lr), "k--")
ax.plot(np.sin(EFE), 0,  "Xr")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-5.5, 6.5])
ax.set_title("(d) Extratropical Transports")
ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")

handles = (l1, l2, l3, l4, l5, l6, l7, l8, l9, l10)
labels = ("$S'$", "$-S\\alpha'$", "$-S'\\alpha$", "$-S'\\alpha'$", "PL", "WV", "LR",  "$NEI'$", "Sum", "EFE")
f.legend(handles, labels, loc="upper center", ncol=5)

plt.tight_layout()
plt.subplots_adjust(top=0.80)

fname = "differences_transports.pdf"
plt.savefig(fname)
print("{} created.".format(fname))
plt.close()
