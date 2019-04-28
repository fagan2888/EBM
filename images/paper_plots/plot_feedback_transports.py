#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc("lines", linewidth=4, markersize=10)
rc("axes", titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, linewidth=1.5)
rc("axes.spines", top=False, right=False)
rc("grid", c="k", ls="--", lw=1, alpha=0.4)
rc("xtick", labelsize=13)
rc("xtick.major", size=5, width=1.5)
rc("ytick", labelsize=13)
rc("ytick.major", size=5, width=1.5)
rc("legend", fontsize=18)

data = np.load("feedback_transports.npz")
delta_flux_total = data["delta_flux_total"]
delta_flux_pl = data["delta_flux_pl"]
delta_flux_wv = data["delta_flux_wv"]
delta_flux_lr = data["delta_flux_lr"]
delta_flux_alb = data["delta_flux_alb"]
delta_S = data["delta_S"]
sin_lats = data["sin_lats"]
EFE = data["EFE"]

f, ax = plt.subplots(1, figsize=(16, 10))
ax.plot(sin_lats, 10**-15 * delta_flux_total, "k", label="Total")
ax.plot(sin_lats, 10**-15 * delta_flux_pl,  "r", label="PL")
ax.plot(sin_lats, 10**-15 * delta_flux_wv, "m", label="WV")
ax.plot(sin_lats, 10**-15 * delta_flux_lr, "y", label="LR")
ax.plot(sin_lats, 10**-15 * delta_flux_alb, "g", label="AL")
ax.plot(sin_lats, 10**-15 * delta_S, "c", label="$\\delta S$")
ax.plot(sin_lats, 10**-15 * (delta_S + delta_flux_pl + delta_flux_wv + delta_flux_lr + delta_flux_alb), "k--", label="$\\delta S + \\sum \\delta F_i$")
ax.plot(np.sin(EFE), 0,  "Xr", label="EFE")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([-6, 6])
ax.set_yticks(np.arange(-6, 7, 1))
ax.grid()
ax.legend(loc="upper left")
ax.set_xlabel("Latitude")
ax.set_ylabel("Transport [PW]")
plt.tight_layout()

fname = "feedback_transports.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
