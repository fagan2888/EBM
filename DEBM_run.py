#!/usr/bin/env python3

################################################################################
### IMPORTS
################################################################################
import numpy as np
from time import clock
from DEBM_lib import *


def run_model():
    print('\nModel Params:')
    print("dtmax:      {:.2f} s / {:.4f} days".format(dtmax, dtmax/60/60/24))
    print("dt:         {:.2f} s / {:.4f} days".format(dt, dt/60/60/24))
    print("dy:         {:.2f} m".format(dy))
    print("iterations: {}".format(int(max_iters)))
    print("tolerance:  {}".format(tolerance*dt))
    print("Time:       {:.2f} s / {:.0f} days".format(t_final, t_final/60/60/24))
    print("frames:     {}\n".format(frames))
    
    print('Insolation Type:   {}'.format(insolation_type))
    if insolation_type == 'perturbed':
        print('\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}'.format(lat0, M, sigma))
    print('Initial Temp Dist: {}'.format(init_condition))
    print('Albedo Feedback:   {}'.format(albedo_feedback))
    print('OLR Scheme:        {}'.format(olr_type))
    if olr_type == 'linear':
        print('\tA = {:.2f}, B = {:.2f}'.format(Acoeff, Bcoeff))
    print('Numerical Method:  {}\n'.format(numerical_method))
    
    T_array = np.zeros((frames, len(lats)))
    E_array = np.zeros((frames, len(lats)))
    alb_array = np.zeros((frames, len(lats)))
    L_array = np.zeros((frames, len(lats)))
    
    T = init_temp
    E = E_dataset[np.searchsorted(T_dataset, T)]
    alb = init_alb
    
    t0 = clock()
    iter_count = 0
    frame_count = 0
    error = -1
    while iter_count < max_iters:
        if iter_count % nPrint == 0: 
            if iter_count == 0:
                print('{:5d}/{:.0f} iterations.'.format(iter_count, max_iters))
            else:
                print('{:5d}/{:.0f} iterations. Last error: {:.16f}'.format(iter_count, max_iters, error))
        if iter_count % Nplot == 0:
            T_array[frame_count, :] = T
            E_array[frame_count, :] = E
            alb_array[frame_count, :] = alb
            L_array[frame_count, :] = L(T)
            error = np.sum(np.abs(T_array[frame_count, :] - T_array[frame_count-1, :]))
            if error < tolerance*dt:
                frame_count += 1
                T_array = T_array[:frame_count, :]
                E_array = E_array[:frame_count, :]
                alb_array = alb_array[:frame_count, :]
                L_array = L_array[:frame_count, :]
                print('{:5d}/{:.0f} iterations. Last error: {:.16f}'.format(iter_count, max_iters, error))
                print('Equilibrium reached in {} iterations ({:.1f} days).'.format(iter_count, iter_count*dt/60/60/24))
                break
            else:
                frame_count += 1
        E, T, alb = take_step(E, T, alb)
        iter_count += 1
    
    tf = clock()
    if T_array.shape[0] == frames:
        print('Failed to reach equilibrium. Final error: {:.16f} K'.format(np.max(np.abs(T_array[-1, :] - T_array[-2, :]))))
    print('\nTime: {:.10f} seconds/iteration\n'.format((tf-t0)/iter_count))
    
    # Save data
    np.savez('data/T_array.npz', T_array)
    np.savez('data/E_array.npz', E_array)
    np.savez('data/L_array.npz', L_array)
    np.savez('data/alb_array.npz', alb_array)
    if olr_type == 'full_wvf':
        prescribed_vapor = state['specific_humidity'].values[:, :, :]
        np.savez('data/prescribed_vapor.npz', prescribed_vapor)


if __name__ == '__main__':
    run_model()
