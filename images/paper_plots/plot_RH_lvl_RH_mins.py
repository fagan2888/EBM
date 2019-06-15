#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

rc("font", size=8)

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

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.47))

p0 = 600
for M in [0, 18]:
    if M != 0:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    else:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]
    Ip0 = np.argmin(np.abs(pressures - p0))
    
    if M != 0:
        ax1.plot(sin_lats, RH[Ip0, :], "m-", label="CESM2 M={} tropical".format(M))
    else:
        ax1.plot(sin_lats, RH[Ip0, :], "k-", label="CESM2 Control")

    if M != 0:
        data = np.load("RH_M{}_mebm.npz".format(M))
    else:
        data = np.load("RH_M{}_mebm.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]/100
    Ip0 = np.argmin(np.abs(pressures - p0))
    
    if M != 0:
        ax1.plot(sin_lats, RH[Ip0, :], "m--", label="MEBM M={} tropical".format(M))
    else:
        ax1.plot(sin_lats, RH[Ip0, :], "k--", label="MEBM Control")

ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax1.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax1.set_ylim([0, 1])
ax1.legend(loc="upper right")
ax1.set_title("(a) RH at {:3d} hPa".format(p0))
ax1.set_xlabel("Latitude")
ax1.set_ylabel("RH")


data = np.loadtxt("data_RHasym.dat")

efes = data[:, 0]
RH_mins_L = data[:, 1]
RH_mins_R = data[:, 2]

RH_ctrl = 1/2*(RH_mins_L[0] + RH_mins_R[0])
m_L, b_L, r_L = linregress(efes, RH_mins_L, b=RH_ctrl)
m_R, b_R, r_R = linregress(efes, RH_mins_R, b=RH_ctrl)
print(m_L, m_R)
print(r_L**2, r_R**2)

xvals = np.linspace(np.min(efes)-10, np.max(efes)+10, 10)

ax2.plot(efes, RH_mins_L, "bo", label="SH minimum")
ax2.plot(xvals, m_L*xvals + RH_ctrl, "b--")
ax2.plot(efes, RH_mins_R, "ro", label="NH minimum")
ax2.plot(xvals, m_R*xvals + RH_ctrl, "r--")

ax2.set_xlim([-12, 1])
ax2.set_xticks(np.arange(-12, 1, 2))
ax2.set_xticklabels(["12°S", "10°S", "8°S", "6°S", "4°S", "2°S", "EQ"])
ax2.set_ylim([0, 0.4])
ax2.set_title("(b) RH Mininima at 600 hPa")
ax2.set_xlabel("EFE Latitude")
ax2.set_ylabel("RH")
ax2.legend()

plt.tight_layout()

fname = "RH_{:3d}_RH_mins.pdf".format(p0)
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
