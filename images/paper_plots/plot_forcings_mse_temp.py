#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

ps = 98000    
cp = 1005     
RH = 0.8      
Lv = 2257000  

def get_dS(perturb_intensity, location):
    if location == "tropics":
        perturb_center = 15
        perturb_spread = 4.94
    elif location == "extratropics":
        perturb_center = 60
        perturb_spread = 9.89
    func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
    perturb_normalizer, er = sp.integrate.quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
    return -perturb_intensity/perturb_normalizer * np.exp(-(np.arcsin(sin_lats) - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))

def humidsat(t, p):
    """
    FROM BOOS:
    % function [esat,qsat,rsat]=_humidsat(t,p)
    %  computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    %  and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    %  pressure (p) in hPa.
    %
    %  these are all computed using the modified Tetens-like formulae given by
    %  Buck (1981, J. Appl. Meteorol.)
    %  for vapor pressure over liquid water at temperatures over 0 C, and for
    %  vapor pressure over ice at temperatures below -23 C, and a quadratic
    %  polynomial interpolation for intermediate temperatures.
    """
    tc=t-273.16;
    tice=-23;
    t0=0;
    Rd=287.04;
    Rv=461.5;
    epsilon=Rd/Rv;

    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    # alternatively don"t use enhancement factor for non-ideal gas correction
    #ewat=6.1121.*exp(17.502.*tc./(240.97+tc));
    #eice=6.1115.*exp(22.452.*tc./(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2

    esat=eint
    esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
    esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]

    # now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    return esat, qsat, rsat

T_dataset = np.arange(33, 350, 1e-3)
q_dataset = humidsat(T_dataset, ps/100)[1]
E_dataset = cp*T_dataset + RH*q_dataset*Lv

# Set up grid
N_pts = 401
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)

# Plot
f, axes = plt.subplots(2, 2, figsize=(7.057, 7.057/1.62))

# top left plot
ax = axes[0, 0]
linestyles = [":", "-.", "--", "-"]
for i, M in enumerate([5, 10, 15, 18]):
    dS = get_dS(M, "tropics")
    ax.plot(sin_lats, dS, "k" + linestyles[i], label="$M={:2d}$".format(M))
ax.annotate("(a)", (0.02, 0.93), xycoords="axes fraction")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-180, 0])
ax.set_xlabel("Latitude")
ax.set_ylabel("Insolation Forcing (W m$^{-2}$)")
ax.grid(False)
ax.legend(loc="lower left")

# top right plot
ax = axes[0, 1]
linestyles = [":", "-.", "--", "-"]
for i, M in enumerate([5, 10, 15, 18]):
    dS = get_dS(M, "extratropics")
    ax.plot(sin_lats, dS, "k" + linestyles[i], label="$M={:2d}$".format(M))
ax.annotate("(b)", (0.02, 0.93), xycoords="axes fraction")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([-180, 0])
ax.set_xlabel("Latitude")
ax.set_ylabel("Insolation Forcing (W m$^{-2}$)")
ax.grid(False)
ax.legend(loc="lower left")

# bottom left plot
ax = axes[1, 0]

directory = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM_sims/sim287"
simulation = "/ctrl"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
ax.plot(sin_lats, T, "k-", label="$M=0$ (control)")

simulation = "/full_radiation/tropical/M15"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
ax.plot(sin_lats, T, "k--", label="$M=15$ tropical")

simulation = "/full_radiation/extratropical/M15"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
ax.plot(sin_lats, T, "k-.", label="$M=15$ extratropical")

ax.annotate("(c)", (0.02, 0.93), xycoords="axes fraction")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel("Latitude")
ax.set_ylabel("Surface Temperature, $T$ (K)")
ax.grid(False)
ax.legend(loc="lower center")

# bottom left plot
ax = axes[1, 1]

directory = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM_sims/sim287"
simulation = "/ctrl"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
E = E_dataset[np.searchsorted(T_dataset, T)]
ax.plot(sin_lats, E / 1000, "k-", label="$M=0$ (control)")

simulation = "/full_radiation/tropical/M15"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
E = E_dataset[np.searchsorted(T_dataset, T)]
ax.plot(sin_lats, E / 1000, "k--", label="$M=15$ tropical")

simulation = "/full_radiation/extratropical/M15"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
E = E_dataset[np.searchsorted(T_dataset, T)]
ax.plot(sin_lats, E / 1000, "k-.", label="$M=15$ extratropical")

ax.annotate("(d)", (0.02, 0.93), xycoords="axes fraction")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel("Latitude")
ax.set_ylabel("Surface MSE, $h$ (kJ kg$^{-1}$)")
ax.grid(False)
ax.legend(loc="lower center")

plt.tight_layout()

fname = "forcings_mse_temp.pdf"
plt.savefig(fname)
print("{} saved.".format(fname))
plt.close()
