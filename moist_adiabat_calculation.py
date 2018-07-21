#!/usr/bin/env python

################################################################################
# A simple script to calculate moist adiabats using metpy
#   for use in the DEBM model.
#
# Henry G. Peterson with Bill Boos -- summer 2018
################################################################################

################################################################################
### IMPORTS
################################################################################
import numpy as np
import climt
from metpy.calc import moist_lapse
from metpy.units import units
from DEBM import RH

# to get nice pressure levels
nLevels = 30
radiation = climt.RRTMGLongwave()
state = climt.get_default_state([radiation], x={}, y={},
                mid_levels={'label' : 'mid_levels', 'values': np.arange(nLevels), 'units' : ''},
                interface_levels={'label' : 'interface_levels', 'values': np.arange(nLevels + 1), 'units' : ''}
                )

################################################################################
### CALCULATION
################################################################################
nSamples = 200
minT = 217    #overflow if minT is below 217 ???
maxT = 350
pressures = state['air_pressure'].values[0, 0, :] * units('Pa') # MetPy requires units
Tsample = np.linspace(minT, maxT, nSamples) * units('K')        # Sample surface temps
Tdata = np.zeros((nSamples, len(pressures))) * units('K')
Tdata[:, 0] = Tsample

print('Calculating moist adiabats...')
for i in range(nSamples):
    if i%10 == 0: print('{}/{}'.format(i, nSamples))
    Tdata[i, :] = moist_lapse(temperature=Tdata[i, 0], pressure=pressures)
print('{}/{}'.format(nSamples, nSamples))

# Keep T constant above 200 hPa for a Tropopause
# Zero RH past this hieght
RH_vals = RH * np.ones(len(pressures))
for i in range(len(pressures)):
    if pressures[i].magnitude/100 < 200:
        Tdata[:, i] = Tdata[:, i-1]
        RH_vals[i] = 0

################################################################################
### DATA SAVING
################################################################################
fname = 'moist_adiabat_data.npz'
np.savez(fname, pressures=pressures, Tsample=Tsample, Tdata=Tdata, RH_vals=RH_vals)
print('Data saved in {}.'.format(fname))

