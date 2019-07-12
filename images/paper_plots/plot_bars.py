#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")
rc("axes", xmargin=0.01)

def linregress(x, y, b=None):
    """
    Compute line of best fit using Normal equations.
    INPUTS
        x: x-values of data points
        y: y-values of data points
        b: requested y-int of line of best fit (if None, calculate)
    OUTPUTS
        m: slope of line of best fit
        b: y-int of line of best fit
        r: coefficient of determination (https://en.wikipedia.org/wiki/Coefficient_of_determination)
    """
    if b == None:
        A = np.zeros((len(x), 2))
        A[:, 0] = x
        A[:, 1] = 1
        ATA = np.dot(A.T, A)
        ATy = np.dot(A.T, y)
    else:
        A = np.zeros((len(x), 1))
        A[:, 0] = x
        ATA = np.dot(A.T, A)
        ATy = np.dot(A.T, y - b)
    
    if b == None:
        [m, b] = np.linalg.solve(ATA, ATy)
    else:
        [m] = np.linalg.solve(ATA, ATy)
    
    y_avg = np.mean(y)
    SS_tot = np.sum((y - y_avg)**2)
    
    f = m*x + b
    SS_reg = np.sum((f - y_avg)**2)
    SS_res = np.sum((y - f)**2)
    
    r = np.sqrt(1 - SS_res / SS_tot)

    return m, b, r


def get_data(filename, location):
    filename   = EBM_PATH + '/data/' + filename
    data_array = np.loadtxt(filename, delimiter=',')
    if location == 'tropics':
        data_array = data_array[np.where(data_array[:, 0] == 15)]
    elif location == 'extratropics':
        data_array = data_array[np.where(data_array[:, 0] == 60)]
    centers = data_array[:, 0]
    spreads = data_array[:, 1]
    intensities = data_array[:, 2]
    efes = data_array[:, 3]
    return centers, spreads, intensities, efes

def integrate_lat(f, i=-1):
    if isinstance(f, np.ndarray):
        if i == -1:
            return  2*np.pi*Re**2 * np.trapz(f, dx=dx) 
        else:
            return  2*np.pi*Re**2 * np.trapz(f[:i+1], dx=dx) 
    else:
        if i == -1:
            return  2*np.pi*Re**2 * np.trapz(f * np.ones(N_pts), dx=dx) 
        else:
            return  2*np.pi*Re**2 * np.trapz(f * np.ones(N_pts)[:i+1], dx=dx) 


def calculate_trans(f, force_zero=False):
    if force_zero:
        area = integrate_lat(1)
        f_bar = 1 / area * integrate_lat(f)
    if isinstance(f, np.ndarray):
        trans = np.zeros(f.shape)
    else:
        trans = np.zeros(N_pts)
    for i in range(N_pts):
        if force_zero:
            trans[i] = integrate_lat(f - f_bar, i)
        else:
            trans[i] = integrate_lat(f, i)
    return trans


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


def get_bar_height(filename, location, control_efes=None):
    centers, spreads, intensities, efes = get_data(filename, location)
    if location == "tropics":
        m, b, r = linregress(efes, dS_eq_trans_t, b=0)
        # m = dS_eq_trans_t[0] / efes[0]
        if isinstance(control_efes, np.ndarray):
            m_ctrl, b_ctrl, r_ctrl = linregress(control_efes, dS_eq_trans_t, b=0)
            # m_ctrl = dS_eq_trans_t[0] / control_efes[0]
            # print(r_ctrl**2)
    elif location == "extratropics":
        m, b, r = linregress(efes, dS_eq_trans_e, b=0)
        # m = dS_eq_trans_e[0] / efes[0]
        if isinstance(control_efes, np.ndarray):
            m_ctrl, b_ctrl, r_ctrl = linregress(control_efes, dS_eq_trans_e, b=0)
            # m_ctrl = dS_eq_trans_e[0] / control_efes[0]
            # print(r_ctrl**2)
    if isinstance(control_efes, np.ndarray):
        height = m_ctrl - m
    else:
        height = m
    if "rh" in filename:
        height *= -1

    print("{:15s} {:5s} {:2.5f} ({:2.1f})".format(labels[i], location[0], height, 1/height))
    # print("{:15s} {:5s} {:2.5f}".format(labels[i], location[0], r**2))
    return height


# Set up arrays
filenames = ["sensitivity_clark.dat", 
            "sensitivity_clark_no_wv.dat", 
            "sensitivity_cesm2.dat", 
            "sensitivity_full_radiation.dat", 
            "sum",
            "sensitivity_full_radiation_no_wv.dat", 
            "sensitivity_full_radiation_no_al.dat", 
            "sensitivity_full_radiation_no_al_no_wv_no_lr.dat", 
            "sensitivity_full_radiation_no_lr.dat", 
            "sensitivity_full_radiation_rh.dat"]
controls = ["None", 
            "sensitivity_clark.dat", 
            "None", 
            "None", 
            "None",
            "sensitivity_full_radiation.dat", 
            "sensitivity_full_radiation.dat", 
            "None", 
            "sensitivity_full_radiation.dat", 
            "sensitivity_full_radiation.dat"] 
labels = ["C18 Total", "C18 WV", "CESM2 Total", "MEBM Total", "MEBM Sum", "MEBM WV", "MEBM AL", "MEBM PL", "MEBM LR", "MEBM RH"]
colors = ["k", "m", "k", "k", "k", "m", "g", "r", "y", "c"]
alphas = [0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
xvals = np.array([0, 1, 3, 5, 6, 7, 8, 9, 10, 11])
hatches = ["/", "/", "\\", "", "", "", "", "", "", ""]

heights_t = np.zeros(len(filenames))
heights_e = np.zeros(len(filenames))

# Set up grid
Re = 6.371e6 
N_pts = 401
dx = 2 / (N_pts - 1)
sin_lats = np.linspace(-1.0, 1.0, N_pts)

# Calculate forcing transports
ctrl_data = np.load(EBM_PATH + "/data/ctrl.npz")
alb_ctrl = ctrl_data["alb"]
dS_eq_trans_t = np.zeros(4)
dS_eq_trans_e = np.zeros(4)
for i, M in enumerate([5, 10, 15, 18]):
    dS = get_dS(M, "tropics")
    dS_trans = calculate_trans(dS*(1 - alb_ctrl), force_zero=True)
    I_equator = N_pts//2
    dS_eq_trans_t[i] = 10**-15 * dS_trans[I_equator] 

    dS = get_dS(M, "extratropics")
    dS_trans = calculate_trans(dS*(1 - alb_ctrl), force_zero=True)
    I_equator = N_pts//2
    dS_eq_trans_e[i] = 10**-15 * dS_trans[I_equator] 

# Plot
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.404, 3.404/1.62*2), sharex=True)

for i, filename in enumerate(filenames):
    if ".dat" in filename:
        control = controls[i]
        if control == "None":
            height_t = get_bar_height(filename, location="tropics", control_efes=None) 
            height_e = get_bar_height(filename, location="extratropics", control_efes=None) 
        else:
            centers, spreads, intensities, control_efes_t = get_data(control, "tropics")
            centers, spreads, intensities, control_efes_e = get_data(control, "extratropics")
            height_t = get_bar_height(filename, location="tropics", control_efes=control_efes_t) 
            height_e = get_bar_height(filename, location="extratropics", control_efes=control_efes_e) 
        heights_t[i] = height_t
        heights_e[i] = height_e
        ax1.bar(xvals[i], height_t, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)
        ax2.bar(xvals[i], height_e, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)

# calculate sum
heights_t[4] = heights_t[5] + heights_t[6] + heights_t[7] + heights_t[8]
heights_e[4] = heights_e[5] + heights_e[6] + heights_e[7] + heights_e[8]
ax1.bar(xvals[4], heights_t[4], color=colors[4], alpha=alphas[4], hatch=hatches[4], align="edge", edgecolor="k", linewidth=0.5)
ax2.bar(xvals[4], heights_e[4], color=colors[4], alpha=alphas[4], hatch=hatches[4], align="edge", edgecolor="k", linewidth=0.5)

ax1.plot([-100, 100], [0, 0], "k-", lw=0.5)
ax2.plot([-100, 100], [0, 0], "k-", lw=0.5)

ax1.annotate("(a)", (0.015, 0.93), xycoords="axes fraction")
ax1.set_ylabel("Feedback Parameter, $\lambda_i$ (PW degrees$^{-1}$)")
ax1.set_ylim([-2.5, 2.0])
ax2.annotate("(b)", (0.015, 0.93), xycoords="axes fraction")
ax2.set_ylabel("Feedback Parameter, $\lambda_i$ (PW degrees$^{-1}$)")
ax2.set_xticks(xvals + 0.4)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_xlim([xvals[0]-0.4, xvals[-1]+1.2])
ax2.set_ylim([-2.5, 2.0])

plt.tight_layout()

fname = 'feedback_param_bars.pdf'
plt.savefig(fname)
print('{} created.'.format(fname))
plt.close()
