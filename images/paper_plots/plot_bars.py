#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")
rc("font", size=8)
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


# filenames = ["sensitivity_clark.dat", "sensitivity_cesm2.dat", "sensitivity_full_radiation.dat", "sensitivity_full_radiation_no_wv.dat", "sensitivity_clark_no_wv.dat", "sensitivity_full_radiation_no_al.dat", "sensitivity_full_radiation_no_lr.dat", "sensitivity_full_radiation_rh.dat", "sensitivity_full_radiation_D_cesm2.dat"]
# labels = ["C18", "CESM", "MEBM", "MEBM No WV", "C18 No WV", "MEBM No AL", "MEBM No LR", "MEBM RH", "CESM $D$"]
# colors = ["k", "k", "k", "m", "k", "g", "y", "c", "r"]
# alphas = [0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
# xvals = np.array([0, 1, 2, 4, 5, 6, 7, 9, 10])
# hatches = ["\\", "", "","","/","","", "", ""]

# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2*2.47), sharex=True)

# for i, filename in enumerate(filenames):
#     location = "tropics"
#     centers, spreads, intensities, efes = get_data(filename, location)
#     # m_t, b_t, r_t = linregress(-intensities, efes)
#     m_t, b_t, r_t = linregress(-intensities, efes, b=0)
#     ax1.bar(xvals[i], m_t, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)

#     location = "extratropics"
#     centers, spreads, intensities, efes = get_data(filename, location)
#     # m_e, b_e, r_e = linregress(-intensities, efes)
#     m_e, b_e, r_e = linregress(-intensities, efes, b=0)
#     ax2.bar(xvals[i], m_e, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)
#     print("{:20s} & {:1.2f} ({:1.4f}) & {:1.2f} ({:1.4f}) \\\\".format(labels[i], m_t, r_t**2, m_e, r_e**2))

# ax1.set_title("(a) Tropical")
# ax1.set_ylabel("EFE Shift [$^\\circ$] per Forcing [W m$^{-2}$]")
# # ax1.set_ylim([0, 0.7])
# ax1.set_ylim([0, 1.0])
# ax1.grid(False)
# ax2.set_title("(b) Extratropical")
# ax2.set_ylabel("EFE Shift [$^\\circ$] per Forcing [W m$^{-2}$]")
# ax2.set_xticks(xvals + 0.4)
# ax2.set_xticklabels(labels, rotation=45, ha="right")
# # ax2.set_ylim([0, 0.7])
# ax2.set_ylim([0, 1.0])
# ax2.grid(False)

# plt.tight_layout()

# fname = 'slopes.pdf'
# plt.savefig(fname)
# plt.show()
# print('{} created.'.format(fname))
# plt.close()

################################################################################
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


def get_bar_height(filename, location, control_efes):
    centers, spreads, intensities, efes = get_data(filename, location)
    if location == "tropics":
        m, b, r = linregress(efes - control_efes, dS_eq_trans_t, b=0)
    elif location == "extratropics":
        m, b, r = linregress(efes - control_efes, dS_eq_trans_e, b=0)
    height = 1 / m
    if "no" in filename:
        height *= -1
    return height

# Set up arrays
filenames = ["sensitivity_clark.dat", "sensitivity_clark_no_wv.dat", "sensitivity_full_radiation.dat", "sensitivity_full_radiation_no_wv.dat", "sensitivity_full_radiation_no_al.dat", "sensitivity_full_radiation_no_lr.dat", "sensitivity_full_radiation_rh.dat"]
controls = [0, 0, 2, 2, 2, 2, 2]
labels = ["C18", "C18 WV", "MEBM", "MEBM WV", "MEBM AL", "MEBM LR", "MEBM RH"]
colors = ["k", "m", "k", "m", "g", "y", "c"]
alphas = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
xvals = np.array([0, 1, 3, 4, 5, 6, 7])
hatches = ["/", "/", "", "", "", "", ""]

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
    dS_trans = calculate_trans(-dS*(1 - alb_ctrl), force_zero=True)
    I_equator = N_pts//2
    dS_eq_trans_t[i] = 10**-15 * dS_trans[I_equator] 

    dS = get_dS(M, "extratropics")
    dS_trans = calculate_trans(-dS*(1 - alb_ctrl), force_zero=True)
    I_equator = N_pts//2
    dS_eq_trans_e[i] = 10**-15 * dS_trans[I_equator] 

# Plot
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2*2.47), sharex=True)

for i, filename in enumerate(filenames):
    control = controls[i]
    if control == i:
        height_t = get_bar_height(filename, location="tropics", control_efes=0) 
        height_e = get_bar_height(filename, location="extratropics", control_efes=0) 
    else:
        centers, spreads, intensities, control_efes_t = get_data(filenames[control], "tropics")
        centers, spreads, intensities, control_efes_e = get_data(filenames[control], "extratropics")
        height_t = get_bar_height(filename, location="tropics", control_efes=control_efes_t) 
        height_e = get_bar_height(filename, location="extratropics", control_efes=control_efes_e) 
    ax1.bar(xvals[i], height_t, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)
    ax2.bar(xvals[i], height_e, color=colors[i], alpha=alphas[i], hatch=hatches[i], align="edge", edgecolor="k", linewidth=0.5)

ax1.plot([-10, 10], [0, 0], "k-")
ax2.plot([-10, 10], [0, 0], "k-")

ax1.set_title("(a) Tropical")
ax1.set_ylabel("Response [degrees PW$^{-1}$]")
ax1.set_ylim([-0.5, 3.5])
ax1.grid(False)
ax2.set_title("(b) Extratropical")
ax2.set_ylabel("Response [degrees PW$^{-1}$]")
ax2.set_xticks(xvals + 0.4)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_xlim([xvals[0]-0.4, xvals[-1]+1.2])
ax2.set_ylim([-0.5, 3.5])
ax2.grid(False)

plt.tight_layout()

fname = 'response_bars.pdf'
plt.savefig(fname)
plt.show()
print('{} created.'.format(fname))
plt.close()
