#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

def plot_diffs(ax):
    l1, = ax.plot(sin_lats, dS*(1 - alb), "c")
    l2, = ax.plot(sin_lats, -(S + dS)*dalb, "g")
    # skip = 10
    # ax.plot(sin_lats[::skip], (-(S + dS)*dalb)[::skip], "go", ms=3)
    l3, = ax.plot(sin_lats, -dL_pl, "r")
    l4, = ax.plot(sin_lats, -dL_wv, "m")
    l5, = ax.plot(sin_lats, -dL_rh, "b")
    l6, = ax.plot(sin_lats, -dL_lr, "y")
    l7, = ax.plot(sin_lats, dS*(1 - alb) - (S + dS)*dalb - dL, "k")
    l8, = ax.plot(sin_lats, dS*(1 - alb) - (S + dS)*dalb - (dL_pl + dL_wv + + dL_rh + dL_lr), "k--")
    l9, = ax.plot(np.sin(EFE), 0,  "Xr")

    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax.set_yticks(np.arange(-150, 101, 50))
    ax.set_ylim([-120, 90])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Energy Perturbation (W m$^{-2}$)")

    return (l1, l2, l3, l4, l5, l6, l7, l8, l9)

def plot_transports(ax):
    ax.plot(sin_lats, 10**-15 * dtrans_dS, "c")
    ax.plot(sin_lats, -10**-15 * dtrans_dalb, "g")
    ax.plot(sin_lats, -10**-15 * dtrans_pl, "r")
    ax.plot(sin_lats, -10**-15 * dtrans_wv, "m")
    ax.plot(sin_lats, -10**-15 * dtrans_rh, "b")
    ax.plot(sin_lats, -10**-15 * dtrans_lr, "y")
    ax.plot(sin_lats, 10**-15 * dtrans_total, "k")
    ax.plot(sin_lats, 10**-15 * (dtrans_dS - dtrans_dalb - (dtrans_pl + dtrans_wv + dtrans_rh + dtrans_lr)), "k--")
    ax.plot(np.sin(EFE), 0,  "Xr")
    
    ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
    ax.set_ylim([-5, 4])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Energy Transport (PW)")

f, axes = plt.subplots(2, 2, figsize=(7.057, 7.057/1.62))

data = np.load("feedback_transports_differences_T15.npz")
EFE = data["EFE"] 
sin_lats = data["sin_lats"] 
dtrans_total = data["dtrans_total"] 
dtrans_pl = data["dtrans_pl"] 
dtrans_wv = data["dtrans_wv"] 
dtrans_rh = data["dtrans_rh"] 
dtrans_lr = data["dtrans_lr"] 
dtrans_dS = data["dtrans_dS"] 
dtrans_dalb = data["dtrans_dalb"] 
dL_pl = data["dL_pl"] 
dL_wv = data["dL_wv"] 
dL_rh = data["dL_rh"] 
dL_lr = data["dL_lr"] 
dL = data["dL"] 
dS = data["dS"]
dalb = data["dalb"]
alb = data["alb"]
S = data["S"]

ax = axes[0, 0]
(l1, l2, l3, l4, l5, l6, l7, l8, l9) = plot_diffs(ax)
# ax.set_title("(a) Tropical Differences")
ax.annotate("(a)", (0.01, 0.92), xycoords="axes fraction")

ax = axes[0, 1]
plot_transports(ax)
# ax.set_title("(b) Tropical Transports")
ax.annotate("(b)", (0.01, 0.92), xycoords="axes fraction")

################################################################################

data = np.load("feedback_transports_differences_E15.npz")
EFE = data["EFE"] 
sin_lats = data["sin_lats"] 
dtrans_total = data["dtrans_total"] 
dtrans_pl = data["dtrans_pl"] 
dtrans_wv = data["dtrans_wv"] 
dtrans_rh = data["dtrans_rh"] 
dtrans_lr = data["dtrans_lr"] 
dtrans_dS = data["dtrans_dS"] 
dtrans_dalb = data["dtrans_dalb"] 
dL_pl = data["dL_pl"] 
dL_wv = data["dL_wv"] 
dL_rh = data["dL_rh"] 
dL_lr = data["dL_lr"] 
dL = data["dL"] 
dS = data["dS"]
dalb = data["dalb"]
alb = data["alb"]
S = data["S"]

ax = axes[1, 0]
(l1, l2, l3, l4, l5, l6, l7, l8, l9) = plot_diffs(ax)

# ax.set_title("(c) Extratropical Differences")
ax.annotate("(c)", (0.01, 0.92), xycoords="axes fraction")

ax = axes[1, 1]
plot_transports(ax)
# ax.set_title("(d) Extratropical Transports")
ax.annotate("(d)", (0.01, 0.92), xycoords="axes fraction")

handles = (l1, l2, l3, l4, l5, l6, l7, l8, l9)
labels = ("$S'(1 - \\alpha)$", "$-(S + S')\\alpha'$", "$-L_{PL}'$", "$-L_{WV}'$", "$-L_{RH}'$", "$-L_{LR}$",  "$NEI'$", "Sum", "EFE")
f.legend(handles, labels, loc="upper center", ncol=9)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

fname = "differences_transports.pdf"
plt.savefig(fname)
print("{} created.".format(fname))
plt.close()
