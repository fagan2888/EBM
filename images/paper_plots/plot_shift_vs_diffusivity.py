#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc("animation", html="html5")
rc("lines", linewidth=6, markersize=15)
rc("axes", titlesize=30, labelsize=25, xmargin=0.01, ymargin=0.01, linewidth=1.5)
rc("axes.spines", top=False, right=False)
rc("grid", c="k", ls="--", lw=1, alpha=0.4)
rc("xtick", labelsize=20)
rc("xtick.major", size=5, width=1.5)
rc("ytick", labelsize=20)
rc("ytick.major", size=5, width=1.5)
rc("legend", fontsize=15)

data = np.loadtxt("analytical_ebm_shifts.dat")
ratios_analytical = data[:, 0]
shifts_analytical = data[:, 1]

data = np.loadtxt("numerical_ebm_shifts.dat")
diffusivity_numerical = data[:, 0]
shifts_numerical = data[:, 1]

lambda_pl = -3.15
lambda_wv = 1.8
lambda_lr = -0.84
Lambda = -(lambda_pl + lambda_wv + lambda_lr)

a = 6.371e6
ps = 98000
g = 9.81
diffusivities = ps/g * diffusivity_numerical
cp = 1005
Lv = 2257000
r = 0.8
Rv = 461.39
Tbasic = 280
Tkelvin = 273.16
beta = Lv/Rv/Tbasic**2
e0 = 0.6112*10**3
R = 287.05
q0 = R*e0/(Rv*ps)
Kappas_numerical = diffusivities*(cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin))) / a**2
Kappa_earth = 2.55e-4 * (cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin)))
Kappa_cesm_min = 2e-5 * (cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin)))
Kappa_cesm_max = 4e-4 * (cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin)))
ratio_earth = Kappa_earth / Lambda
ratio_cesm_min = Kappa_cesm_min / Lambda
ratio_cesm_max = Kappa_cesm_max / Lambda
ratios_numerical = Kappas_numerical / Lambda

f, ax = plt.subplots(1, figsize=(16, 10))
ax.plot([ratio_earth, ratio_earth], [-1e10, 1e10], "r--", label="CESM Average")
ax.plot([ratio_cesm_min, ratio_cesm_min], [-1e10, 1e10], "r--", alpha=0.5, label="CESM Min/Max")
ax.plot([ratio_cesm_max, ratio_cesm_max], [-1e10, 1e10], "r--", alpha=0.5)
ax.plot(ratios_analytical, shifts_analytical, "k-", label="Analytical Model")
ax.plot(ratios_numerical, shifts_numerical, "bo", label="Numerical Model")
ax.set_xlim([-0.1, 2.6])
ax.set_ylim([-14, 0])
ax.grid()
ax.legend(loc="upper right")
ax.set_xlabel("$\\kappa / \\Lambda$")
ax.set_ylabel("EFE shift [degrees]")
plt.tight_layout()

fname = "shift_vs_diffusivity.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
