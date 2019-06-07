#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

EBM_PATH = os.environ["EBM_PATH"]

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


filenames = {'sensitivity_clark.dat' : "Clark et al.",
            'sensitivity_clark_no_wv.dat' : "Clark et al. No WV",
            'sensitivity_cesm2.dat' : "CESM2",
            'sensitivity_full_radiation.dat' : "MEBM",
            'sensitivity_full_radiation_no_al.dat' : "MEBM No AL",
            'sensitivity_full_radiation_no_wv.dat' : "MEBM No WV",
            'sensitivity_full_radiation_no_lr.dat' : "MEBM No LR",
            'sensitivity_full_radiation_rh.dat' : "MEBM RH",
            'sensitivity_full_radiation_D_cesm2.dat' : "CESM2 $D$",
            'sensitivity_full_radiation_D1.dat' : "$D_1$",
            'sensitivity_full_radiation_D2.dat' : "$D_2$"}
xvals = np.linspace(0, 20, 100)
for filename in filenames:
    location = "tropics"
    centers, spreads, intensities, efes = get_data(filename, location)
    m_t, b_t, r_t = linregress(intensities, efes)
    # plt.plot(intensities, efes, "bo")
    # plt.plot(xvals, xvals*m_t + b_t, "b-")
    location = "extratropics"
    centers, spreads, intensities, efes = get_data(filename, location)
    m_e, b_e, r_e = linregress(intensities, efes)
    # plt.plot(intensities, efes, "ro")
    # plt.plot(xvals, xvals*m_e + b_e, "r-")
    print("{:20s} & {:1.2f} ({:1.4f}) & {:1.2f} ({:1.4f}) \\\\".format(filenames[filename], m_t, r_e**2, m_e, r_t**2))
    # print(b_t, b_e)
    # plt.show()
