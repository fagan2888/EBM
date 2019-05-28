#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

cesm2_data = np.load(EBM_PATH + "/data/D_cesm2.npz")
sin_lats_cesm2 = cesm2_data["sin_lats"]
D_cesm2 = cesm2_data["D"]

ratio = 10/16
width = 16
height = ratio*width
f, ax = plt.subplots(1, figsize=(width, height))
ax.plot(sin_lats_cesm2, D_cesm2, "k", label="CESM2")
ax.plot([-1, 1], [2.6e-4, 2.6e-4], "k--", label="Hwang & Frierson 2010")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
ax.set_ylim([0, 5e-4])
ax.grid()
ax.legend(loc="upper left")
ax.set_title("Diffusivity", pad=10)
ax.set_xlabel("Latitude")
ax.set_ylabel("$D$ [kg m$^{-2}$ s$^{-1}$]")

plt.tight_layout()

fname = "diffusivity.png"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
