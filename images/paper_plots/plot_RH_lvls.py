#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

####

f, ax = plt.subplots(1, figsize=(16, 10))

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
        ax.plot(sin_lats, RH[Ip0, :], "r-", label="CESM2 M={} tropical".format(M))
    else:
        ax.plot(sin_lats, RH[Ip0, :], "k-", label="CESM2 Control")

    if M != 0:
        data = np.load("RH_M{}_mebm.npz".format(M))
    else:
        data = np.load("RH_M{}_mebm.npz".format(M))
    RH = data["RH"]
    sin_lats = np.sin(data["lats"])
    pressures = data["pressures"]/100
    Ip0 = np.argmin(np.abs(pressures - p0))
    
    if M != 0:
        ax.plot(sin_lats, RH[Ip0, :], "r--", label="MEBM M={} tropical".format(M))
    else:
        ax.plot(sin_lats, RH[Ip0, :], "k--", label="MEBM Control")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
ax.set_ylim([0, 1])
ax.grid()
ax.legend(loc="lower left")
ax.set_xlabel("Latitude")
ax.set_ylabel("RH (at {:3d} mb)".format(p0))
plt.tight_layout()

fname = "RH_{:3d}.png".format(p0)
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))

#####

#f, ax = plt.subplots(1, figsize=(16, 10))

#for M in [0, 1, 3, 5, 10, 15, 18]:
#    data = np.load("RH_M{}_cesm2.npz".format(M))
#    RH = data["RH"]
#    sin_lats = np.sin(data["lats"])
#    pressures = data["pressures"]
#    I500 = np.argmin(np.abs(pressures - 500))
    
#    if M != 0:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={} tropical".format(M))
#    else:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

#ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
#ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
#ax.set_ylim([0, 1])
#ax.grid()
#ax.legend(loc="lower left")
#ax.set_xlabel("Latitude")
#ax.set_ylabel("RH (at 500 mb)")
#plt.tight_layout()

#fname = "RH500.png"
#plt.savefig(fname)
## plt.show()

#print("{} saved.".format(fname))
#plt.close()

#####

#f, ax = plt.subplots(1, figsize=(16, 10))

#for M in [0, 5, 10, 15, 18]:
#    if M != 0:
#        data = np.load("RH_M{}extrop_cesm2.npz".format(M))
#    else:
#        data = np.load("RH_M{}_cesm2.npz".format(M))
#    RH = data["RH"]
#    sin_lats = np.sin(data["lats"])
#    pressures = data["pressures"]
#    I500 = np.argmin(np.abs(pressures - 500))
    
#    if M != 0:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={} extratropical".format(M))
#    else:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

#ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
#ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
#ax.set_ylim([0, 1])
#ax.grid()
#ax.legend(loc="lower left")
#ax.set_xlabel("Latitude")
#ax.set_ylabel("RH (at 500 mb)")
#plt.tight_layout()

#fname = "RH500extrop.png"
#plt.savefig(fname)
## plt.show()

#print("{} saved.".format(fname))
#plt.close()

#####

#f, ax = plt.subplots(1, figsize=(16, 10))

#for M in [0, 18]:
#    if M != 0:
#        data = np.load("RH_M{}_cesm2.npz".format(M))
#    else:
#        data = np.load("RH_M{}_cesm2.npz".format(M))
#    RH = data["RH"]
#    sin_lats = np.sin(data["lats"])
#    pressures = data["pressures"]
#    I500 = np.argmin(np.abs(pressures - 500))
    
#    if M != 0:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={} tropical".format(M))
#    else:
#        ax.plot(sin_lats, RH[I500, :], label="CESM M={}".format(M))

#for M in [0, 18]:
#    if M != 0:
#        data = np.load("RH_M{}_mebm.npz".format(M))
#    else:
#        data = np.load("RH_M{}_mebm.npz".format(M))
#    RH = data["RH"]
#    sin_lats = np.sin(data["lats"])
#    pressures = data["pressures"]/100
#    I500 = np.argmin(np.abs(pressures - 500))
    
#    if M != 0:
#        ax.plot(sin_lats, RH[I500, :], label="MEBM M={} tropical".format(M))
#    else:
#        ax.plot(sin_lats, RH[I500, :], label="MEBM M={}".format(M))

#ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
#ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
#ax.set_ylim([0, 1])
#ax.grid()
#ax.legend(loc="lower left")
#ax.set_xlabel("Latitude")
#ax.set_ylabel("RH (at 500 mb)")
#plt.tight_layout()

#fname = "RH500mebm.png"
#plt.savefig(fname)
#plt.show()

#print("{} saved.".format(fname))

#####

#f, ax = plt.subplots(1, figsize=(16, 10))

#for M in [0]:
#    if M != 0:
#        data = np.load("RH_M{}_cesm2.npz".format(M))
#    else:
#        data = np.load("RH_M{}_cesm2.npz".format(M))
#    RH = data["RH"]
#    sin_lats = np.sin(data["lats"])
#    pressures = data["pressures"]
#    for pres in [950, 500, 150]:
#        I = np.argmin(np.abs(pressures - pres))
        
#        if M != 0:
#            ax.plot(sin_lats, RH[I, :], label="CESM M={} tropical, p={:4.0f} hPa".format(M, pres))
#        else:
#            ax.plot(sin_lats, RH[I, :], label="CESM M={}, p={:4.0f} hPa".format(M, pres))

#ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
#ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
#ax.set_ylim([0, 1])
#ax.grid()
#ax.legend(loc="lower left")
#ax.set_xlabel("Latitude")
#ax.set_ylabel("RH")
#plt.tight_layout()

#fname = "RHlevels.png"
#plt.savefig(fname)
#plt.show()

#print("{} saved.".format(fname))
