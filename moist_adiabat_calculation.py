#!/usr/bin/env python

################################################################################
# A simple script to calculate moist adiabats using metpy
#   for use in the EBM.
#
# Henry G. Peterson with Bill Boos 2019
################################################################################

################################################################################
### IMPORTS
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import climt
from metpy.calc import moist_lapse
from metpy.units import units
from metpy.plots import SkewT
from EBM import RH

# to get same pressure levels as CliMT
N_levels = 30
grid = climt.get_grid(nx=1, ny=1, nz=N_levels)
state = climt.get_default_state([], grid_state=grid)
pressures = state['air_pressure'].values[:,0,0] 

# setup sample Ts points
N_sample_pts = 20    # test
# N_sample_pts = 200    # full run
minT = 217    #overflow if minT is below 217 ???
maxT = 350
T_surf_sample = np.linspace(minT, maxT, N_sample_pts) 

T_data = np.zeros((N_sample_pts, N_levels)) 
T_data[:, 0] = T_surf_sample

print('Calculating moist adiabats...')
for i in range(N_sample_pts):
    if i%10 == 0: print(i)
    T_data[i, :] = moist_lapse(temperature=T_data[i, 0]*units('K'), pressure=pressures*units('Pa'))

# Keep T constant above 200 hPa for a Tropopause
for i in range(len(pressures)):
    if pressures[i]/100 < 200:
        T_data[:, i] = T_data[:, i-1]

# Debug plots:
f = plt.figure(figsize=(9, 9))
skew = SkewT(f, rotation=45)
for i in range(N_sample_pts):
    skew.plot(pressures/100, T_data[i, :] - 273.15, 'r')
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(minT - 273.15, maxT - 273.15)
skew.plot_moist_adiabats()
plt.show()

# # Save data
# fname = 'data/moist_adiabat_data.npz'
# np.savez(fname, pressures=pressures, T_surf_sample=T_surf_sample, T_data=T_data)
# print('Data saved in {}.'.format(fname))

