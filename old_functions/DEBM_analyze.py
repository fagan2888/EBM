#!/usr/bin/env python3

################################################################################
### IMPORTS AND STYLES
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from matplotlib import animation, rc
from DEBM_lib import *

rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.05, ymargin=0.05, linewidth=1.5)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=5, width=1.5)
rc('ytick', labelsize=13)
rc('ytick.major', size=5, width=1.5)
rc('legend', fontsize=14)


################################################################################
### LOAD DATA
################################################################################
T_array   = np.load('data/T_array.npz')['arr_0']
E_array   = np.load('data/E_array.npz')['arr_0']
L_array   = np.load('data/L_array.npz')['arr_0']
alb_array = np.load('data/alb_array.npz')['arr_0']

################################################################################
### INITIAL DISTRIBUTIONS
################################################################################
print('\nPlotting Initial Dists')
fig, ax = plt.subplots(1, figsize=(12,5))

# radiaiton dist
SW = S*(1 - init_alb)
LW = L(init_temp)
ax.plot([-90, 90], [0, 0], 'k--', lw=2)
ax.plot(lats, SW, 'r', lw=2, label='SW (with albedo)', alpha=0.5)
ax.plot(lats, LW, 'g', lw=2, label='LW (init)', alpha=0.5)
ax.plot(lats, SW - LW, 'b', lw=2, label='SW - LW', alpha=1.0)

ax.set_title('SW/LW Radiation (init)')
ax.set_xlabel('Lat')
ax.set_ylabel('W/m$^2$')
ax.legend(loc='upper right')

plt.tight_layout()

fname = 'init_rad_dists.png'
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()

################################################################################
### FINAL TEMP DIST
################################################################################
print('\nPlotting Final T Dist')
f, ax = plt.subplots(1, figsize=(12,5))
ax.plot(lats, T_array[-1, :], 'k')
ax.set_title("Final Temperature Distribution")
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('T (K)')
ax.grid(c='k', ls='--', lw=1, alpha=0.4)
ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

plt.tight_layout()

fname = 'final_temp_dists.png'
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()

print('Mean T: {:.2f} K'.format(np.mean(T_array[-1,:])))

# Planck
# Mean T: 289.0272833342359 K 

# A1
# Mean T: 292.4582833342523 K

# A2
# Mean T: 289.38176666757096 K


################################################################################
### FIND ITCZ
################################################################################
print('\nPlotting EFE')
E_f = E_array[-1, :] / 1000

spl = UnivariateSpline(lats, E_f, k=4, s=0)
roots = spl.derivative().roots()

max_index = np.argmax(E_f)
efe_lat = lats[max_index]

min_error_index = np.argmin( np.abs(roots - efe_lat) )
closest_root = roots[min_error_index]

print("Approximate EFE lat = {:.1f}".format(efe_lat))
print("Root found = {:.16f} (of {})".format(closest_root, len(roots)))
print("ITCZ = {:.5f}".format(closest_root*0.64))

f, ax = plt.subplots(1, figsize=(12,5))
ax.plot(lats, E_f, 'c', label='Final Energy Distribution', lw=4)
ax.plot(lats, spl(lats), 'k--', label='Spline Interpolant')
min_max = [E_f.min(), E_f.max()]
ax.plot([efe_lat, efe_lat], min_max, 'm')
ax.plot([closest_root, closest_root], min_max, 'r')
ax.text(efe_lat+5, np.average(min_max), "EFE $\\approx$ {:.2f}$^\\circ$".format(closest_root), size=16)
ax.set_title("Final Energy Distribution")
ax.legend(fontsize=14, loc="upper left")
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('E (kJ / kg)')
ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

plt.tight_layout()

fname = 'efe.png'
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()


################################################################################
### FINAL RADIATION DIST
################################################################################
print('\nPlotting Final Radiation Dist')
f, ax = plt.subplots(1, figsize=(12, 5))

T_f = T_array[-1, :]
alb_f = alb_array[-1, :]
SW_f = S * (1 - alb_f)
LW_f = L(T_f)
ax.plot(lats, SW_f, 'r', label='S(1-$\\alpha$)')
ax.plot(lats, LW_f, 'b', label='OLR')
ax.plot(lats, SW_f - LW_f, 'g', label='Net')
ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax.grid()
ax.legend(loc='upper left')
ax.set_title("Final Radiation Distributions")
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("W/m$^2$")

plt.tight_layout()

fname = 'final_rad_dist.png'
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()

print('(SW - LW) at EFE: {:.2f} W/m^2'.format(SW_f[max_index] - LW_f[max_index]))


################################################################################
### LW vs. T
################################################################################
print('\nPlotting LW vs. T')
f, ax = plt.subplots(1, figsize=(8, 5))
ax.set_title("Relationship Between T$_s$ and OLR")
ax.set_xlabel("T$_s$ (K)")
ax.set_ylabel("OLR (W/m$^2$)")

def f(t, a, b):
    return a + b*t

x_data = T_array.flatten()
y_data = L_array.flatten()

popt, pcov = curve_fit(f, x_data, y_data)
print('A: {:.2f} W/m2, B: {:.2f} W/m2/K'.format(popt[0], popt[1]))

xvals = np.linspace(np.min(x_data), np.max(x_data), 1000)
yvals = f(xvals, *popt)

ax.plot(x_data, y_data, 'co', ms=2, label='data points: "{}"'.format(olr_type))
ax.plot(xvals, f(xvals, *popt), 'k--', label='linear fit')
ax.text(np.min(xvals), np.mean(yvals), s='A = {:.2f},\nB = {:.2f}'.format(popt[0], popt[1]), size=14)
ax.legend()

plt.tight_layout()

fname = 'OLR_vs_T_fit_{}.png'.format(olr_type)
plt.savefig(fname, dpi=120)
print('{} created.'.format(fname))
plt.close()

# Linear OLR: It works!
# A: -281.6700000000001 W/m2, B: 1.8000000000000003 W/m2/K

# PLANCK OLR:
# A: -652.8783145676047 W/m2, B: 3.1022145805413275 W/m2/K

# Full OLR, WV Feedback:
# A: -417.0478973801873 W/m2, B: 2.349441658553002 W/m2/K

# Full OLR, No WV Feedback:
# A: -383.62200068534526 W/m2, B: 2.2236068235897157 W/m2/K
# A: -418.26464942476156 W/m2, B: 2.3554184845083475 W/m2/K    (starting at T_f from WV feedback)

# Shell Somerville:
# A: -999.1626913063144 W/m2, B: 4.033450693435474 W/m2/K      (k = 0.03)
# A: -1417.4283920090295 W/m2, B: 5.681866740765462 W/m2/K     (k = 0.20)


################################################################################
### ANIMATION
################################################################################
show = 'T'
# show = 'E'
# show = 'alb'
# show = 'L'
print('\nCreating Animation of {}'.format(show))

# set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(1, figsize=(9,5))

ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax.set_xlabel('Latitude (degrees)')
if show == 'T':
    array = T_array
    ax.set_ylabel('T (K)')
elif show == 'E':
    array = E_array
    ax.set_ylabel("W/m$^2$")
elif show == 'alb':
    array = alb_array
    ax.set_ylabel("$\\alpha$")
elif show == 'L':
    array = L_array
    ax.set_ylabel("W/m$^2$")
ax.set_title('EBM t =  0 days')
plt.tight_layout(pad=3)

line, = ax.plot(lats, array[0, :], 'b')

def init():
    line.set_data(lats, array[0, :])
    return (line,)

def animate(i):
    if i%100 == 0: 
        print("{}/{} frames".format(i, len(T_array)))
    ax.set_title('EBM t = {:.0f} days'.format((i+1)*Nplot*dt/60/60/24))
    graph = array[i, :]
    line.set_data(lats, graph)
    m = graph.min()
    M = graph.max()
    ax.set_ylim([m - 0.01*np.abs(m), M + 0.01*np.abs(M)])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(T_array), interval=int(anim_length/len(T_array)), blit=True)

fname = '{}_anim.mp4'.format(show)
anim.save(fname)
print('{} created.'.format(fname))

#################################################################################
#### VERTICAL AIR TEMP
#################################################################################
#for i in range(nLevels):
#    plt.plot(lats, state['air_temperature'].values[0, :, i], 'k-', lw=1.0)
#plt.plot(lats, state['air_temperature'].values[0, :, 0], 'r-')
#plt.xlabel('latitude')
#plt.ylabel('temp')
#print(state['air_temperature'])
#plt.figure()
#lat = 45
#plt.plot(state['air_temperature'].values[0, int(lat/dlat), :], pressures/100, 'k-', label='{}$^\\circ$ lat'.format(lat))
#plt.gca().invert_yaxis()
#plt.legend()
#plt.xlabel('temp')
#plt.ylabel('pressure')
#
#
#################################################################################
#### VERTICAL MOISTURE
#################################################################################
#for i in range(nLevels):
#    plt.plot(lats, state['specific_humidity'].values[0, :, i], 'k-', lw=1.0)
#plt.plot(lats, state['specific_humidity'].values[0, :, 0], 'r-')
#plt.xlabel('latitude')
#plt.ylabel('sp hum')
#print(state['specific_humidity'])
#plt.figure()
#lat = 45
#plt.plot(state['specific_humidity'].values[0, int(lat/dlat), :], pressures/100, 'k-', label='{}$^\\circ$ lat'.format(lat))
#plt.gca().invert_yaxis()
#plt.legend()
#plt.xlabel('sp hum')
#plt.ylabel('pressure')
