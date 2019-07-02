#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

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

a = 6.371e6

# Set up grid
N_pts = 401
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)
yvals = a*np.arcsin(sin_lats)

# Plotting and calculating 
f, ax = plt.subplots(1)
for location in ["tropics", "extratropics"]:
    for M in [5, 10, 15, 18]:
        # Set up dS
        dS = get_dS(M, location)

        # Set up line
        if location == "tropics":
            color = "r"
        else:
            color = "b"
        if M == 5:
            linestyle = "-"
        if M == 10:
            linestyle = "--"
        if M == 15:
            linestyle = "-."
        if M == 18:
            linestyle = ":"
        
        # Set up Kappa and Lambda
        ctrl_data = np.load(EBM_PATH + "/data/ctrl.npz")
        
        ps = 98000
        g = 9.81
        D   = 1.06e6
        diffusivity = ps/g * D
        cp = 1005
        Lv = 2257000
        r = 0.8
        Rv = 461.39
        e0 = 0.6112*10**3
        R = 287.05
        Tkelvin = 273.16
        
        Lambda = -2.19
        
        Tbasics = np.linspace(50, 320, 50)
        dT_maxes = np.zeros(Tbasics.shape)
        Kappas = np.zeros(Tbasics.shape)
        
        for i, Tbasic in enumerate(Tbasics):
            beta = Lv/Rv/Tbasic**2
            
            q0 = R*e0/(Rv*ps)
            
            Kappa = diffusivity*(cp + Lv*r*beta*q0*np.exp(beta*(Tbasic - Tkelvin))) 
            Kappas[i] = Kappa
            gamma = np.sqrt(-Lambda/Kappa)
            
            # Calculate dT as convolution with Green's function
            dT = np.zeros(N_pts)
            for j in range(N_pts):
                # Greens = -1/(2*np.roll(Kappa*gamma, N_pts//2)) * np.exp(-np.roll(gamma, N_pts//2)*np.abs(yvals - yvals[i]))
                Greens = -1/(2*Kappa*gamma) * np.exp(-gamma*np.abs(yvals - yvals[j]))
                dT[j] = np.trapz(Greens * -dS, x=(yvals - yvals[j]))
            dT_maxes[i] = np.max(np.abs(dT))


        print(M, location)
        ax.plot(Kappas / a**2, dT_maxes, "{}{}".format(color, linestyle), label="M={:2d}, {}".format(M, location)) 

ax.set_xlabel("$\\kappa$ [kg K$^{-1}$ s$^{-3}$]")
ax.set_ylabel("Max |$T'$| [K]")
ax.set_title("Predicting Maximum $T'$ Given $\\kappa$")
ax.legend()

plt.tight_layout()

fname = "dT_vs_kappa.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
