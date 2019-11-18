#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.integrate, scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mebm
import os

# EBM_PATH = os.environ["EBM_PATH"]
EBM_PATH = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM"
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

def calculate_efe(M, location):
    filename = EBM_PATH + '/data/sensitivity_full_radiation.dat' 
    data_array = np.loadtxt(filename, delimiter=',')
    if location == 'tropics':
        data_array = data_array[np.where(data_array[:, 0] == 15)]
    elif location == 'extratropics':
        data_array = data_array[np.where(data_array[:, 0] == 60)]
    EFE = data_array[np.where(data_array[:, 2] == M), 3]
    return np.deg2rad(EFE[0][0])

T_dataset = np.arange(33, 350, 1e-3)
q_dataset = humidsat(T_dataset, ps/100)[1]
E_dataset = cp*T_dataset + RH*q_dataset*Lv

# Set up grid
N_pts = 513
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)

# Plot
m = mebm.MoistEnergyBalanceModel(N_pts=513)
ctrl_alb = m.ctrl_data["alb"]

f, axes = plt.subplots(3, 1, figsize=(3.404, 3.404/1.62*3))

ax = axes[0]
for i, M in enumerate([5, 10, 15, 18]):
    dS = get_dS(M, "tropics")
    ax.plot(sin_lats, dS*(1-ctrl_alb), "r-", alpha=0.5, label="$M={:2d}$".format(M))
    dS = get_dS(M, "extratropics")
    ax.plot(sin_lats, dS*(1-ctrl_alb), "b--", alpha=0.5)
dS = 0 * dS
ax.plot(sin_lats, dS, color="k", linestyle="-", label="$M=0$, control", lw=1.5)

ax.annotate("(a)", (0.02, 0.94), xycoords="axes fraction")
ax.annotate("control", xy=(np.sin(np.deg2rad(15.5)),   0), xycoords="data", xytext=(-80, -12), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
ax.annotate("$M=5$", xy=(np.sin(np.deg2rad(14.5)),   0.8*-47),  xycoords="data", xytext=(-45, 0.8*-1.8), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate("$M=10$", xy=(np.sin(np.deg2rad(14.5)),  0.8*-95),  xycoords="data", xytext=(-45, 0.8*-1.8), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate("$M=15$", xy=(np.sin(np.deg2rad(14.5)),  0.8*-140), xycoords="data", xytext=(-45, 0.8*-1.8), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.annotate("$M=18$", xy=(np.sin(np.deg2rad(14.5)),  0.8*-170), xycoords="data", xytext=(-45, 0.8*-1.8), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_ylim([0.8*-200, 0.8*20])
ax.set_xlabel("Latitude")
ax.set_ylabel("Forcing, $S'(1 - \\alpha)$ (W m$^{-2}$)")

legend_elements = [Line2D([0], [0], color="r", linestyle="-",  alpha=0.5, label="Tropical"),
                   Line2D([0], [0], color="b", linestyle="--", alpha=0.5, label="Extratropical")]
ax.legend(handles=legend_elements, loc="lower left")

ax = axes[1]

T_min = 205
T_max = 301

directory = "/home/hpeter/Documents/ResearchBoos/EBM_files/EBM_sims/sim292"
for i, M in enumerate([5, 10, 15, 18]):
    simulation = "/full_radiation/T{:02d}".format(M)
    data = np.load(directory + simulation + "/simulation_data.npz")
    T = data["T"][-1, :]
    ax.plot(sin_lats, T, "r-", alpha=0.5)
    EFE = calculate_efe(M, "tropics")
    ax.plot([np.sin(EFE), np.sin(EFE)], [T_min, np.max(T)], "r-", alpha=0.5)

    simulation = "/full_radiation/E{:02d}".format(M)
    data = np.load(directory + simulation + "/simulation_data.npz")
    T = data["T"][-1, :]
    ax.plot(sin_lats, T, "b--", alpha=0.5)
    EFE = calculate_efe(M, "extratropics")
    ax.plot([np.sin(EFE), np.sin(EFE)], [T_min, np.max(T)], "b--", alpha=0.5)
simulation = "/ctrl"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
ax.plot(sin_lats, T, "k-", lw=1.5)
EFE = 0
ax.plot([np.sin(EFE), np.sin(EFE)], [T_min, np.max(T)], "k-")

ax.annotate("(b)", (0.02, 0.94), xycoords="axes fraction")
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel("Latitude")
ax.set_ylabel("Surface Temperature, $T$ (K)")
ax.set_ylim([T_min, T_max])

ax = axes[2]

E_min = 210
E_max = 342

for i, M in enumerate([5, 10, 15, 18]):
    simulation = "/full_radiation/T{:02d}".format(M)
    data = np.load(directory + simulation + "/simulation_data.npz")
    T = data["T"][-1, :]
    E = E_dataset[np.searchsorted(T_dataset, T)]
    ax.plot(sin_lats, E / 1000, "r-", alpha=0.5)
    EFE = calculate_efe(M, "tropics")
    ax.plot([np.sin(EFE), np.sin(EFE)], [E_min, np.max(E)/1000], "r-", alpha=0.5)

    simulation = "/full_radiation/E{:02d}".format(M)
    data = np.load(directory + simulation + "/simulation_data.npz")
    T = data["T"][-1, :]
    E = E_dataset[np.searchsorted(T_dataset, T)]
    ax.plot(sin_lats, E / 1000, "b--", alpha=0.5)
    EFE = calculate_efe(M, "extratropics")
    ax.plot([np.sin(EFE), np.sin(EFE)], [E_min, np.max(E)/1000], "b--", alpha=0.5)
simulation = "/ctrl"
data = np.load(directory + simulation + "/simulation_data.npz")
T = data["T"][-1, :]
E = E_dataset[np.searchsorted(T_dataset, T)]
EFE = 0
ax.plot(sin_lats, E / 1000, color="k", linestyle="-", lw=1.5)
ax.plot([np.sin(EFE), np.sin(EFE)], [E_min, np.max(E)/1000], "k-")

ax.annotate("(c)", (0.02, 0.94), xycoords="axes fraction")
ax.annotate("EFEs", xy=(0, np.mean([E_min, E_max])), xycoords="data", xytext=(20, -10), textcoords="offset points", 
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.set_xlabel("Latitude")
ax.set_ylabel("Surface MSE, $h$ (kJ kg$^{-1}$)")
ax.set_ylim([E_min, E_max])

plt.tight_layout()

fname = "forcings_mse_temp.pdf"
plt.savefig(fname)
print(fname)
plt.close()

