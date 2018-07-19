#!/usr/bin/env python3

################################################################################
### IMPORTS 
################################################################################
import numpy as np
from scipy.interpolate import UnivariateSpline
from DEBM_lib import *
from DEBM_run import run_model

def get_EFE():
    '''
    Calculate EFE from last E_array data
    '''
    # Get data and final dist
    E_array = np.load('data/E_array.npz')['arr_0']
    E_f = E_array[-1, :] / 1000
    
    # Interp and find roots
    spl = UnivariateSpline(lats, E_f, k=4, s=0)
    roots = spl.derivative().roots()
    
    # Find supposed root based on actual data
    max_index = np.argmax(E_f)
    efe_lat = lats[max_index]
    
    # Pick up closest calculated root to the supposed one
    min_error_index = np.argmin( np.abs(roots - efe_lat) )
    closest_root = roots[min_error_index]

    return closest_root

run_model()
efe = get_EFE()

print('{:2d}, {:2.5f}'.format(M, efe))
