#!/usr/bin/env python3

################################################################################
### IMPORTS 
################################################################################
import numpy as np
from scipy.optimize import curve_fit

################################################################################
### LOAD DATA
################################################################################
T_array   = np.load('data/T_array.npz')['arr_0']
E_array   = np.load('data/E_array.npz')['arr_0']
L_array   = np.load('data/L_array.npz')['arr_0']
alb_array = np.load('data/alb_array.npz')['arr_0']

################################################################################
### LW vs. T
################################################################################
def f(t, a, b):
    return a + b*t

x_data = T_array.flatten()
y_data = L_array.flatten()

popt, pcov = curve_fit(f, x_data, y_data)
print('A: {:.2f} W/m2, B: {:.2f} W/m2/K'.format(popt[0], popt[1]))

# PLANCK OLR:
# A: -652.8783145676047 W/m2, B: 3.1022145805413275 W/m2/K

# Full OLR, WV Feedback:
# A: -378.78 W/m2, B: 2.21 W/m2/K    # from an extratropical perturbation sim
# A: -412.05 W/m2, B: 2.33 W/m2/K    # annual_mean_clark

# Full OLR, No WV Feedback:
# A: -383.62200068534526 W/m2, B: 2.2236068235897157 W/m2/K
# A: -418.26464942476156 W/m2, B: 2.3554184845083475 W/m2/K    (starting at T_f from WV feedback)

# Shell Somerville:
# A: -999.1626913063144 W/m2, B: 4.033450693435474 W/m2/K      (k = 0.03)
# A: -1417.4283920090295 W/m2, B: 5.681866740765462 W/m2/K     (k = 0.20)
