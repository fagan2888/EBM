#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

data = np.loadtxt("sensitivity_cesm2.dat", delimiter=',')
EFEs = data[:, 3]
ITCZs = data[:, 4]

slope, intercept, r_value, p_value, std_err = linregress(EFEs, ITCZs)
print("m={}, b={}".format(slope, intercept))
print("r^2={}".format(r_value**2))

xvals = np.linspace(np.min(EFEs), np.max(EFEs), 1000)

f, ax = plt.subplots(1, figsize=(16,10))

ax.plot(EFEs, ITCZs, "bo")
ax.plot(xvals, slope*xvals + intercept, "k--")

ax.grid()

plt.show()
