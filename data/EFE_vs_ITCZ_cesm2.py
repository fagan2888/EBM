#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")


def linregress(x, y):
    """
    Compute line of best fit using Normal equations.
    INPUTS
        x: x-values of data points
        y: y-values of data points
    OUTPUTS
        m: slope of line of best fit
        b: y-int of line of best fit
        r: coefficient of determination (https://en.wikipedia.org/wiki/Coefficient_of_determination)
    """
    A = np.zeros((len(x), 2))
    A[:, 0] = x
    A[:, 1] = 1
    ATA = np.dot(A.T, A)
    ATy = np.dot(A.T, y)
    
    [m, b] = np.linalg.solve(ATA, ATy)
    
    y_avg = np.mean(y)
    SS_tot = np.sum((y - y_avg)**2)
    
    f = m*x + b
    SS_reg = np.sum((f - y_avg)**2)
    SS_res = np.sum((y - f)**2)
    
    r = np.sqrt(1 - SS_res / SS_tot)

    return m, b, r

data = np.loadtxt("sensitivity_cesm2.dat", delimiter=',')
EFEs = data[:, 3]
ITCZs = data[:, 4]

m, b, r = linregress(EFEs, ITCZs)

print("m   =", m)
print("b   =", b)
print("r^2 =", r**2)


xvals = np.linspace(np.min(EFEs), np.max(EFEs), 1000)

f, ax = plt.subplots(1)

ax.plot(EFEs, ITCZs, "bo")
ax.plot(xvals, m*xvals + b, "k--")

plt.show()
