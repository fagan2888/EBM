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

####

f, ax = plt.subplots(1, figsize=(16, 10))

for M in [0, 1, 3, 5, 10, 15, 18]:
    data = np.load("RH_M{}_cesm2.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]
    I500 = np.argmin(np.abs(pressures - 500))
    
    if M != 0:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={} tropical".format(M))
    else:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([0, 1])
ax.grid()
ax.legend(loc="lower left")
ax.set_xlabel("Latitude")
ax.set_ylabel("RH (at 500 mb)")
plt.tight_layout()

fname = "RH500.png"
plt.savefig(fname)
# plt.show()

print("{} saved.".format(fname))
plt.close()

####

f, ax = plt.subplots(1, figsize=(16, 10))

for M in [0, 5, 10, 15, 18]:
    if M != 0:
        data = np.load("RH_M{}extrop_cesm2.npz".format(M))
    else:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]
    I500 = np.argmin(np.abs(pressures - 500))
    
    if M != 0:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={} extratropical".format(M))
    else:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([0, 1])
ax.grid()
ax.legend(loc="lower left")
ax.set_xlabel("Latitude")
ax.set_ylabel("RH (at 500 mb)")
plt.tight_layout()

fname = "RH500extrop.png"
plt.savefig(fname)
# plt.show()

print("{} saved.".format(fname))
plt.close()

####

f, ax = plt.subplots(1, figsize=(16, 10))

for M in [0, 18]:
    if M != 0:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    else:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]
    I500 = np.argmin(np.abs(pressures - 500))
    
    if M != 0:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={} tropical".format(M))
    else:
        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

for M in [0, 18]:
    if M != 0:
        data = np.load("RH_M{}_mebm.npz".format(M))
    else:
        data = np.load("RH_M{}_mebm.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]/100
    I500 = np.argmin(np.abs(pressures - 500))
    
    if M != 0:
        ax.plot(sin_lats, RH[I500, :], label="MEBM M={} tropical".format(M))
    else:
        ax.plot(sin_lats, RH[I500, :], label="MEBM M={}".format(M))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([0, 1])
ax.grid()
ax.legend(loc="lower left")
ax.set_xlabel("Latitude")
ax.set_ylabel("RH (at 500 mb)")
plt.tight_layout()

fname = "RH500mebm.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))

####

f, ax = plt.subplots(1, figsize=(16, 10))

for M in [0]:
    if M != 0:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    else:
        data = np.load("RH_M{}_cesm2.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]
    for pres in [950, 500, 150]:
        I = np.argmin(np.abs(pressures - pres))
        
        if M != 0:
            ax.plot(sin_lats, RH[I, :], label="CESM M={} tropical, p={:4.0f} hPa".format(M, pres))
        else:
            ax.plot(sin_lats, RH[I, :], label="CESM M={}, p={:4.0f} hPa".format(M, pres))

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
ax.set_ylim([0, 1])
ax.grid()
ax.legend(loc="lower left")
ax.set_xlabel("Latitude")
ax.set_ylabel("RH")
plt.tight_layout()

fname = "RHlevels.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
