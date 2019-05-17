#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc("animation", html="html5")
rc("lines", linewidth=4, markersize=10)
rc("axes", titlesize=30, labelsize=25, xmargin=0.01, ymargin=0.01, linewidth=1.5)
rc("axes.spines", top=False, right=False)
rc("grid", c="k", ls="--", lw=1, alpha=0.4)
rc("xtick", labelsize=20)
rc("xtick.major", size=5, width=1.5)
rc("ytick", labelsize=20)
rc("ytick.major", size=5, width=1.5)
rc("legend", fontsize=15)

E2 = np.load("E_D2.npz")["arr_0"][-1, :]
E3 = np.load("E_D3.npz")["arr_0"][-1, :]
E4 = np.load("E_D4.npz")["arr_0"][-1, :]
E5 = np.load("E_D5.npz")["arr_0"][-1, :]
E6 = np.load("E_D6.npz")["arr_0"][-1, :]
E7 = np.load("E_D7.npz")["arr_0"][-1, :]
E8 = np.load("E_D8.npz")["arr_0"][-1, :]
E9 = np.load("E_D9.npz")["arr_0"][-1, :]

N_pts = E2.shape[0]
dx = 2 / N_pts
sin_lats = np.linspace(-1.0 + dx/2, 1.0 - dx/2, N_pts)

a = 6.371e6
ps = 98000
g = 9.81

D2 = ps / g / a**2 * 1e2
D3 = ps / g / a**2 * 1e3
D4 = ps / g / a**2 * 1e4
D5 = ps / g / a**2 * 1e5
D6 = ps / g / a**2 * 1e6
D7 = ps / g / a**2 * 1e7
D8 = ps / g / a**2 * 1e8
D9 = ps / g / a**2 * 1e9

f, ax = plt.subplots(1, figsize=(16, 10))
ax.plot(sin_lats, 10**-3 * E2, "k", alpha=1.0, label="D = {:1.1E}".format(D2))
ax.plot(sin_lats, 10**-3 * E3, "k", alpha=0.9, label="D = {:1.1E}".format(D3))
ax.plot(sin_lats, 10**-3 * E4, "k", alpha=0.8, label="D = {:1.1E}".format(D4))
ax.plot(sin_lats, 10**-3 * E5, "k", alpha=0.7, label="D = {:1.1E}".format(D5))
ax.plot(sin_lats, 10**-3 * E6, "r", alpha=1.0, label="D = {:1.1E}".format(D6))
ax.plot(sin_lats, 10**-3 * E7, "k", alpha=0.5, label="D = {:1.1E}".format(D7))
ax.plot(sin_lats, 10**-3 * E8, "k", alpha=0.4, label="D = {:1.1E}".format(D8))
ax.plot(sin_lats, 10**-3 * E9, "k", alpha=0.3, label="D = {:1.1E}".format(D9))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([150, 400])
ax.grid()
ax.legend(loc="upper left")
ax.set_xlabel("Latitude")
ax.set_ylabel("MSE [kJ / kg]")
plt.tight_layout()

fname = "MSE_vs_diffusivity.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
