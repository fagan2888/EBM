#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

# Load data
data = np.load("simulation_data_constant.npz")
sin_lats = data["sin_lats"]
trans_constant = data["trans_total"]
data = np.load("simulation_data_D1.npz")
trans_D1 = data["trans_total"]
data = np.load("simulation_data_D2.npz")
trans_D2 = data["trans_total"]
data = np.load("simulation_data_cesm2.npz")
trans_cesm = data["trans_total"]

# Calc derivatives
spl = sp.interpolate.UnivariateSpline(sin_lats, trans_constant, k=4, s=0)
trans_constant_der = spl.derivative()(0)
spl = sp.interpolate.UnivariateSpline(sin_lats, trans_D1, k=4, s=0)
trans_D1_der = spl.derivative()(0)
spl = sp.interpolate.UnivariateSpline(sin_lats, trans_D2, k=4, s=0)
trans_D2_der = spl.derivative()(0)
spl = sp.interpolate.UnivariateSpline(sin_lats, trans_cesm, k=4, s=0)
trans_cesm_der = spl.derivative()(0)

print("Derivatives at EQ:")
print("\tConstant D: {:1.5E}".format(trans_constant_der))
print("\tD1:         {:1.5E}".format(trans_D1_der))
print("\tD2:         {:1.5E}".format(trans_D2_der))
print("\tCESM D:     {:1.5E}".format(trans_cesm_der))

f, ax = plt.subplots(1)

ax.plot(sin_lats, 10**-15 * trans_constant, "k", label="Constant $D$")
ax.plot(sin_lats, 10**-15 * trans_D1, "b", label="$D_1$")
ax.plot(sin_lats, 10**-15 * trans_D2, "g", label="$D_2$")
ax.plot(sin_lats, 10**-15 * trans_cesm, "r", label="CESM $D$")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.legend(loc="lower right")

ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")
ax.set_title("Basic State Transports")

plt.tight_layout()

fname = "basic_trans.png"
plt.savefig(fname)
print("{} created.".format(fname))
plt.close()

################################################################################

# Load data
data = np.load("simulation_data_constant_T5.npz")
trans_constant_T5 = data["trans_total"]
data = np.load("simulation_data_D1_T5.npz")
trans_D1_T5 = data["trans_total"]
data = np.load("simulation_data_D2_T5.npz")
trans_D2_T5 = data["trans_total"]
data = np.load("simulation_data_cesm2_T5.npz")
trans_cesm_T5 = data["trans_total"]

data = np.load("simulation_data_constant_E5.npz")
trans_constant_E5 = data["trans_total"]
data = np.load("simulation_data_D1_E5.npz")
trans_D1_E5 = data["trans_total"]
data = np.load("simulation_data_D2_E5.npz")
trans_D2_E5 = data["trans_total"]
data = np.load("simulation_data_cesm2_E5.npz")
trans_cesm_E5 = data["trans_total"]

f, ax = plt.subplots(1)

# ax.plot(sin_lats, 10**-15 * (trans_constant_T5 - trans_constant), "k", label="Constant $D$")
# ax.plot(sin_lats, 10**-15 * (trans_D1_T5 - trans_D1), "b", label="$D_1$")
# ax.plot(sin_lats, 10**-15 * (trans_D2_T5 - trans_D2), "g", label="$D_2$")
# ax.plot(sin_lats, 10**-15 * (trans_cesm_T5 - trans_cesm), "r", label="CESM $D$")
ax.plot(sin_lats, 10**-15 * (trans_constant_E5 - trans_constant), "k", label="Constant $D$")
ax.plot(sin_lats, 10**-15 * (trans_D1_E5 - trans_D1), "b", label="$D_1$")
ax.plot(sin_lats, 10**-15 * (trans_D2_E5 - trans_D2), "g", label="$D_2$")
ax.plot(sin_lats, 10**-15 * (trans_cesm_E5 - trans_cesm), "r", label="CESM $D$")

ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
ax.legend(loc="upper left")

ax.set_xlabel("Latitude")
ax.set_ylabel("Energy Transport [PW]")
ax.set_title("Anomalous Transports")

plt.tight_layout()

fname = "anom_trans.png"
plt.savefig(fname)
print("{} created.".format(fname))
plt.close()
