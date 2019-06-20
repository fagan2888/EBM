#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

EBM_PATH = os.environ["EBM_PATH"]
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")
rc("font", size=8)
rc("axes", xmargin=0.01)

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


files = {'sensitivity_clark.dat':                  {"label" : "C18", "color" : "k", "alpha" : 0.5, "xpos" : 0, "hatch" : "\\"},
         'sensitivity_cesm2.dat':                  {"label" : "CESM2", "color" : "k", "alpha" : 0.5, "xpos" : 1, "hatch" : ""},
         'sensitivity_full_radiation.dat':         {"label" : "MEBM","color" : "k", "alpha" : 1.0, "xpos" : 2, "hatch" : ""},

         'sensitivity_full_radiation_no_wv.dat':   {"label" : "MEBM No WV","color" : "m", "alpha" : 1.0, "xpos" : 4, "hatch" : ""},
         'sensitivity_clark_no_wv.dat':            {"label" : "C18 No WV", "color" : "k", "alpha" : 0.5, "xpos" : 5, "hatch" : "/"},
         'sensitivity_full_radiation_no_al.dat':   {"label" : "MEBM No AL","color" : "g", "alpha" : 1.0, "xpos" : 6, "hatch" : ""},
         'sensitivity_full_radiation_no_lr.dat':   {"label" : "MEBM No LR","color" : "y", "alpha" : 1.0, "xpos" : 7, "hatch" : ""},

         'sensitivity_full_radiation_rh.dat':      {"label" : "MEBM RH","color" : "c", "alpha" : 1.0, "xpos" : 9, "hatch" : ""},
         'sensitivity_full_radiation_D_cesm2.dat': {"label" : "CESM2 $D$","color" : "r", "alpha" : 1.0, "xpos" : 10, "hatch" : ""}
         # 'sensitivity_full_radiation_D1.dat':      {"label" : "$D_1$","color" : "b", "alpha" : 1.0, "xpos" : 0, "hatch" : ""},
         # 'sensitivity_full_radiation_D2.dat':      {"label" : "$D_2$", "color" : "g", "alpha" : 1.0, "xpos" : 0, "hatch" : ""}
}
labels = []
xticks = []
for d in files.values():
    labels.append(d["label"])
    xticks.append(d["xpos"] + 0.4)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2*2.47), sharex=True)

for filename in files:
    location = "tropics"
    centers, spreads, intensities, efes = get_data(filename, location)
    # m_t, b_t, r_t = linregress(-intensities, efes)
    m_t, b_t, r_t = linregress(-intensities, efes, b=0)
    ax1.bar(files[filename]["xpos"], m_t, color=files[filename]["color"], alpha=files[filename]["alpha"], hatch=files[filename]["hatch"], align="edge", edgecolor="k", linewidth=0.5)

    location = "extratropics"
    centers, spreads, intensities, efes = get_data(filename, location)
    # m_e, b_e, r_e = linregress(-intensities, efes)
    m_e, b_e, r_e = linregress(-intensities, efes, b=0)
    ax2.bar(files[filename]["xpos"], m_e, color=files[filename]["color"], alpha=files[filename]["alpha"], hatch=files[filename]["hatch"], align="edge", edgecolor="k", linewidth=0.5)
    print("{:20s} & {:1.2f} ({:1.4f}) & {:1.2f} ({:1.4f}) \\\\".format(files[filename]["label"], m_t, r_t**2, m_e, r_e**2))

ax1.set_title("(a) Tropical")
ax1.set_ylabel("EFE Shift [$^\\circ$] per Forcing [W m$^{-2}$]")
# ax1.set_ylim([0, 0.7])
ax1.set_ylim([0, 1.0])
ax1.grid(False)
ax2.set_title("(b) Extratropical")
ax2.set_ylabel("EFE Shift [$^\\circ$] per Forcing [W m$^{-2}$]")
ax2.set_xticks(xticks)
ax2.set_xticklabels(labels, rotation=45, ha="right")
# ax2.set_ylim([0, 0.7])
ax2.set_ylim([0, 1.0])
ax2.grid(False)

plt.tight_layout()

fname = 'slopes.pdf'
plt.savefig(fname)
plt.show()
print('{} created.'.format(fname))
plt.close()
