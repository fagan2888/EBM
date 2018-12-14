#!/usr/bin/env python

################################################################################
### IMPORTS
################################################################################
import numpy as np
from scipy.integrate import quadrature, trapz
from scipy.interpolate import UnivariateSpline

################################################################################
### CONSTANTS 
################################################################################
ps  = 98000    #Pa = kg/m/s2
cp  = 1005     #J/kg/K
g   = 9.81     #m/s2
D   = 1.06e6   #m2/s
Re  = 6.371e6  #m
RH  = 0.8      #0-1
S0  = 1365     #J/m2/s
R   = 287.058  #J/kg/K
Lv  = 2257000  #J/kg
sig = 5.67e-8  #J/s/m2/K4

L_f = np.load('L_array.npz')['arr_0'][-1, :]
E_f = np.load('E_array.npz')['arr_0'][-1, :]
T_f = np.load('T_array.npz')['arr_0'][-1, :]
alb_f = np.load('alb_array.npz')['arr_0'][-1, :]
N = L_f.shape[0]

dlat     = 180 / (N - 1)
dlat_rad = np.deg2rad(dlat)
lats     = np.linspace(-90 + dlat/2, 90 - dlat/2, int(180/dlat) + 1)
lats_rad = np.deg2rad(lats)
cos_lats = np.cos(lats_rad)
sin_lats = np.sin(lats_rad)

perturb_center = 15
perturb_spread = 4.94
perturb_intensity = 5
get_S_control = lambda lat: S0*np.cos(np.deg2rad(lat))/np.pi
func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
perturb_normalizer, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
get_dS = lambda lat: - perturb_intensity/perturb_normalizer * np.exp(-(lat - perturb_center)**2 / (2*perturb_spread**2))
get_S = lambda lat: get_S_control(lat) + get_dS(lat)
S_f = (1 - alb_f) * get_S(lats)

### EFE
# Interp and find roots
spl = UnivariateSpline(lats, E_f, k=4, s=0)
roots = spl.derivative().roots()

# Find supposed root based on actual data
max_index = np.argmax(E_f)
efe_lat = lats[max_index]

# Pick up closest calculated root to the supposed one
min_error_index = np.argmin( np.abs(roots - efe_lat) )
closest_root = roots[min_error_index]

EFE = closest_root
EFE_rad = np.deg2rad(EFE)
print()
print('EFE: {:.2E} deg'.format(EFE))

### FEEDBACKS
# Total
flux_total = (2 * np.pi * Re * cos_lats) * (- ps / g * D / Re) * np.gradient(E_f, dlat_rad)

# Planck
emissivity = 0.6
L_planck = emissivity * sig * T_f**4

L_avg = trapz( L_planck * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

flux_planck = np.zeros( L_planck.shape )
for i in range(L_planck.shape[0]):
    flux_planck[i] = trapz( (L_planck[:i+1] - L_avg) * 2 * np.pi * Re**2 * cos_lats[:i+1], dx=dlat_rad)

# Water Vapor 1
L_f_const_q = np.load('data/L_array_constant_q_steps_mid_level_gaussian5_n361.npz')['arr_0'][-1, :]

L_avg = trapz( (L_f - L_f_const_q) * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

flux_wv1 = np.zeros( L_f.shape )
for i in range(L_f.shape[0]):
    flux_wv1[i] = trapz( ( (L_f[:i+1] - L_f_const_q[:i+1]) - L_avg) * 2 * np.pi * Re**2 * cos_lats[:i+1], dx=dlat_rad)

# Water Vapor 2
L_f_zero_q = np.load('L_f_zero_q.npz')['arr_0']

L_avg = trapz( (L_f - L_f_zero_q) * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

flux_wv2 = np.zeros( L_f.shape )
for i in range(L_f.shape[0]):
    flux_wv2[i] = trapz( ( (L_f[:i+1] - L_f_zero_q[:i+1]) - L_avg) * 2 * np.pi * Re**2 * cos_lats[:i+1], dx=dlat_rad)

#####

L_avg = trapz( L_f * 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad) / trapz( 2 * np.pi * Re**2 * cos_lats, dx=dlat_rad)

I = lats.shape[0]//2 + 1
integral_NEI = trapz( (S_f[:I] - L_f[:I]) * 2 * np.pi * Re**2 * cos_lats[:I], dx=dlat_rad )

integral_L = trapz( (L_f[:I] - L_avg) * 2 * np.pi * Re**2 * cos_lats[:I], dx=dlat_rad )

F = UnivariateSpline(lats_rad, flux_total, k=4, s=0)

print('Integral S - L from SP to EQ: {:.2E} W/m'.format(integral_NEI))
print('F(0):                         {:.2E} W/m'.format(F(0)))

print('EFE approx with NEI:  {:.2E} deg'.format( np.rad2deg(- integral_NEI / F.derivative()(EFE_rad)) ))
print('EFE approx with F(0): {:.2E} deg'.format( np.rad2deg(- F(0) / F.derivative()(EFE_rad))) )

integral_NEI = trapz( (S_f[:I] - L_avg) * 2 * np.pi * Re**2 * cos_lats[:I], dx=dlat_rad )

def calc_shift(fluxes=[]):
    if len(fluxes) == 0:
        shift = - integral_NEI / F.derivative()(EFE_rad)
    else:
        numerator = integral_NEI
        denominator = -F.derivative()(EFE_rad)
        for flux in fluxes:
            spl = UnivariateSpline(lats_rad, flux, k=4, s=0)
            numerator -= spl(EFE_rad)
            denominator -= spl.derivative()(EFE_rad)
        shift = numerator / denominator
    return np.rad2deg(shift)

no_fb     = calc_shift()
planck_fb = calc_shift([flux_planck])
wv_fb1    = calc_shift([flux_wv1])
wv_fb2    = calc_shift([flux_wv2])
planck_and_wv_fb1    = calc_shift([flux_planck, flux_wv1])
planck_and_wv_fb2    = calc_shift([flux_planck, flux_wv2])

print()
print('no_fb:                 {:.2E} deg'.format(no_fb))
print('planck_fb:             {:.2E} deg'.format(planck_fb))
print('wv_fb1:                {:.2E} deg'.format(wv_fb1))
print('wv_fb2:                {:.2E} deg'.format(wv_fb2))
print('planck_fb + wv_fb1:    {:.2E} deg'.format(planck_and_wv_fb1))
print('planck_fb + wv_fb2:    {:.2E} deg'.format(planck_and_wv_fb2))


# import matplotlib.pyplot as plt
# from matplotlib import rc

# rc('animation', html='html5')
# rc('lines', linewidth=2, color='b', markersize=10)
# rc('axes', titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, 
#         linewidth=1.5)
# rc('axes.spines', top=False, right=False)
# rc('xtick', labelsize=13)
# rc('xtick.major', size=5, width=1.5)
# rc('ytick', labelsize=13)
# rc('ytick.major', size=5, width=1.5)
# rc('legend', fontsize=14)

# f, ax = plt.subplots(1, figsize=(16, 10))
# # ax.plot(sin_lats, spl(lats_rad), 'k-', label='total flux')
# ax.plot(sin_lats, spl.derivative()(lats_rad), 'b-', label='divergence of total flux')
# ax.plot(sin_lats, 10**-15 * (S_f - L_f) * 2 * np.pi * Re**2 * cos_lats, 'r-', label='NEI')
# ax.legend()
# ax.grid()
# plt.show()
