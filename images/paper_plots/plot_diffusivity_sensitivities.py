#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from plot_sensitivities import get_data
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

rc("font", size=10)

cesm2_data = np.load(EBM_PATH + "/data/D_cesm2.npz")
sin_lats_cesm2 = cesm2_data["sin_lats"]
D_cesm2 = cesm2_data["D"]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.47), gridspec_kw={"width_ratios": [2, 1, 1]})

ax1.plot(sin_lats_cesm2, D_cesm2, "k", label="CESM2")
ax1.plot([-1, 1], [2.6e-4, 2.6e-4], "k--", label="Hwang & Frierson 2010")

ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax1.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax1.set_ylim([0, 5e-4])
ax1.legend(loc="upper left")
ax1.set_title("(a) Diffusivities")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("$D$ [kg m$^{-2}$ s$^{-1}$]")

location = "tropics"
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation.dat", location)
ax2.plot(intensities, efes, marker="o", color="k", linestyle='', label="Constant $D$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D_cesm2.dat", location)
ax2.plot(intensities, efes, marker="P", color="r", linestyle='', label="CESM $D$")

ax2.set_xlim(0, 20)
ax2.set_xticks([5, 10, 15, 18])
ax2.set_ylim(-16, 0)
ax2.set_yticks(np.arange(-16, 1, 2))
ax2.set_yticklabels(['16°S', '14°S', '12°S', '10°S', '8°S', '6°S', '4°S', '2°S', 'EQ'])
ax2.set_title('(b) Tropics')
ax2.set_xlabel('M [W m$^{-2}$]')
ax2.set_ylabel('EFE Latitude')

location = "extratropics"
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation.dat", location)
ax3.plot(intensities, efes, marker="o", color="k", linestyle='', label="Constant $D$")
centers, spreads, intensities, efes = get_data("sensitivity_full_radiation_D_cesm2.dat", location)
ax3.plot(intensities, efes, marker="P", color="r", linestyle='', label="CESM $D$")

ax3.set_xlim(0, 20)
ax3.set_xticks([5, 10, 15, 18])
ax3.set_ylim(-16, 0)
ax3.set_yticks(np.arange(-16, 1, 2))
ax3.set_yticklabels(['', '', '', '', '', '', '', '', ''])
ax3.set_title('(c) Extratropics')
ax3.set_xlabel('M [W m$^{-2}$]')
ax3.legend(loc="lower left")

plt.tight_layout()

fname = "diffusivity_sensitivities.pdf"
plt.savefig(fname)
plt.show()

print("{} saved.".format(fname))
