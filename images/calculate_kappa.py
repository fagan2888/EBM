#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

ctrl_data = np.load(EBM_PATH + "/data/ctrl.npz")
N_pts = 401
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)

a = 6.371e6
ps = 98000
g = 9.81
D   = 1.06e6
diffusivity = ps/g * D / a**2
cp = 1005
Lv = 2257000
r = 0.8
Rv = 461.39
e0 = 0.6112*10**3
R = 287.05

Tbasic = ctrl_data["ctrl_state_temp"][0, :, 0]
Tkelvin = 273.16

beta = Lv/Rv/Tbasic**2

q0 = R*e0/(Rv*ps)


Kappa = diffusivity*(cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin))) 

f, ax = plt.subplots(1)

ax.plot(sin_lats, Kappa, "k-")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel("Latitude")
ax.set_ylabel("$\\kappa$ [kg K$^{-1}$ s$^{-3}$]")
ax.set_title("$\\kappa$ for MEBM Control Simulation")

plt.tight_layout()

fname = "kappa.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
