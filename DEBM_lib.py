#!/usr/bin/env python3

################################################################################
### IMPORTS
################################################################################
import numpy as np
import climt
from metpy.calc import moist_lapse
from metpy.units import units
from scipy.integrate import quadrature
from scipy.interpolate import interp2d


################################################################################
### CONSTANTS 
################################################################################
ps = 98000    #Pa = kg/m/s2
cp = 1005     #J/kg/K
g = 9.81      #m/s2
D = 1.06e6    #m2/s
Re = 6.371e6  #m 
RH = 0.8      #0-1 
S0 = 1365     #J/m2/s
R = 287.058   #J/kg/K
Lv = 2257000  #J/kg
sig = 5.67e-8 #J/s/m2/K4

# numerical variables
# dlat = 0.5
dlat = 1.5
dy = np.pi*Re*dlat/180
dtmax = 0.5*dy**2/D
# dt = dtmax
dt = 1.0 * dtmax
# max_iters = 5e4
max_iters = 1e4
tolerance = 0.01 # K
t_final = max_iters*dt
# Nplot = 500
Nplot = 100
frames = int(t_final / dt / Nplot)
anim_length = 8000

# xvals --- important to have set up early
lats = np.linspace(-90, 90, int(180/dlat))


################################################################################
# INSOLATION
################################################################################
# insolation_type = 'annual_mean'
insolation_type = 'annual_mean_clark'
# insolation_type = 'perturbed'
# insolation_type = 'summer_mean'

if insolation_type == 'annual_mean':
    obliquity = np.deg2rad(23.4)
    eccentricity = 0.0167
    Q0 = S0/4
    Q = Q0 / np.sqrt(1 - eccentricity**2)
    p2 = lambda y: (3*y**2 - 1)/2
    p4 = lambda y: (35*y**4 - 30*y**2 + 3)/8
    p6 = lambda y: (231*y**6 - 315*y**4 + 105*y**2 - 5)/16
    cosB = np.cos(obliquity)
    s_approx = lambda y: 1 - 5/8*p2(cosB)*p2(y) - 9/64*p4(cosB)*p4(y) - 65/1024*p6(cosB)*p6(y)
    get_S = lambda lat: Q * s_approx(np.sin(np.deg2rad(lat)))
elif insolation_type == 'annual_mean_clark':
    get_S = lambda lat: S0*np.cos(np.deg2rad(lat))/np.pi
elif insolation_type == 'perturbed':
    get_S_control = lambda lat: S0*np.cos(np.deg2rad(lat))/np.pi
    ############################################################################
    ###CHANGE VALS
    lat0 = 15; sigma = 4.94
#     lat0 = 60; sigma = 9.89
#     M = 5
#     M = 10
#     M = 15
    M = 18
    ###
    ############################################################################
    func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(lat0))**2 / (2*np.deg2rad(sigma)**2)) * np.cos(y)
    M0, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
    get_dS = lambda lat: - M/M0 * np.exp(-(lat - lat0)**2 / (2*sigma**2))
    get_S = lambda lat: get_S_control(lat) + get_dS(lat)
elif insolation_type == 'summer_mean':
    dec = np.deg2rad(23.45/2)
    get_h0 = lambda lat: np.arccos( -np.tan(np.deg2rad(lat)) * np.tan(dec))
    def get_S(lat_array):
        ss = np.zeros(len(lat_array))
        for i in range(len(lat_array)):
            lat = lat_array[i]
            if lat >= 90 - np.rad2deg(dec):
                h0 = np.pi
            elif lat <= -90 + np.rad2deg(dec):
                h0 = 0
            else:
                h0 = get_h0(lat)
            s = S0 / np.pi * (h0*np.sin(np.deg2rad(lat))*np.sin(dec) + np.cos(np.deg2rad(lat))*np.cos(dec)*np.sin(h0))
            ss[i] = s
        return ss
    
S = get_S(lats)


################################################################################
### HUMIDITY
################################################################################
def humidsat(t,p):
    """
    FROM BOOS:
    % function [esat,qsat,rsat]=humidsat(t,p)
    %  computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    %  and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    %  pressure (p) in hPa.
    %
    %  these are all computed using the modified Tetens-like formulae given by
    %  Buck (1981, J. Appl. Meteorol.)
    %  for vapor pressure over liquid water at temperatures over 0 C, and for
    %  vapor pressure over ice at temperatures below -23 C, and a quadratic
    %  polynomial interpolation for intermediate temperatures.
    """
    tc=t-273.16;
    tice=-23;
    t0=0;
    Rd=287.04;
    Rv=461.5;
    epsilon=Rd/Rv;

    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    # alternatively don't use enhancement factor for non-ideal gas correction
    #ewat=6.1121.*exp(17.502.*tc./(240.97+tc));
    #eice=6.1115.*exp(22.452.*tc./(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2

    esat=eint
    esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
    esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]

    # now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    return esat, qsat, rsat


################################################################################
# DATASETS 
################################################################################
T_dataset = np.arange(100, 400, 1e-3)
q_dataset = humidsat(T_dataset, ps/100)[1]
E_dataset = cp*T_dataset + RH*q_dataset*Lv


################################################################################
# FEEDBACKS
################################################################################
# albedo_feedback = True
albedo_feedback = False


################################################################################
# ALBEDO
################################################################################
if albedo_feedback == True:
    a1 = 0.8
    a2 = 0.1 
    init_alb = 0*lats + a2
    init_alb[np.where(init_temp <= 273.16)] = a1
else:
#     init_alb = 0.2725 * np.ones(len(lats))
    init_alb = (0.3129-0.1959) * np.ones(len(lats))
    
# 67 / 342
# .19590643274853801169

# 107 / 342
# .31286549707602339181

alb = init_alb


################################################################################
# INITIAL CONDITIONS
################################################################################
# init_condition = 'parabola'
# init_condition = 'cos2'
init_condition = 'triangle'
# init_condition = 'top-hat'

if init_condition == 'parabola':
    f0 = 270
    f1 = 305
    f2 = 270
    init_temp = f0 + (f1-f0)/90*(lats + 90) + (f2-2*f1+f0)/90/180*(lats + 90)*(lats + 0)
elif init_condition == 'cos2':
    init_temp = 270 + 35 * np.cos(np.deg2rad(lats))**2
elif init_condition == 'triangle':
    init_temp = (305 - 35/90*np.abs(lats))
#     init_temp = (305 - 35/90*np.abs(lats))
elif init_condition == 'top-hat':
    init_temp = 0*lats + 270
    init_temp[int(1/3*len(lats)):int(2/3*len(lats))] = 305
    
# init_temp = T_f


################################################################################
# OLR
################################################################################
# olr_type = 'planck'
# olr_type = 'full_wvf'
# olr_type = 'full_no_wvf'
olr_type = 'linear'
# olr_type = 'shell_somerville'

if olr_type == 'planck':
    ''' PLANCK RADIATION '''
    # optical_depth = 1.5
    # integrand = lambda tau: tau**(4*R/cp) * np.exp(-tau)
    # integral, error = quadrature(integrand, 0, optical_depth, tol=1e-16, rtol=1e-16, maxiter=1000)
    # L = lambda T: (1-alb)*sig*T**4*np.exp(-optical_depth) + sig*T**4 * optical_depth**(-4*R/cp) * integral #J/m2/s
    # print("Equivalent emissivity: {:.2f}"
    #       .format((1-alb[0])*np.exp(-optical_depth)+optical_depth**(-4*R/cp) * integral))
    emis = 0.6
    L = lambda T: emis*sig*T**4  #J/m2/s
elif olr_type == 'linear':
    ''' LINEAR FIT '''
    lambda_planck = - 3.10 #W/m2/K
    lambda_water  = + 1.80
    lambda_clouds = + 0.68
    lambda_albedo = + 0.26
    lambda_lapse  = - 0.84
    # just planck
    Bcoeff = - (lambda_planck)
    Acoeff = -652.8783145676047
    # planck + water + lapse = full_wvf
    Bcoeff = - (lambda_planck + lambda_water + lambda_lapse)
    # planck + lapse = full_no_wvf
    Bcoeff = - (lambda_planck + lambda_lapse)
#     Acoeff = -281.67
#     Boeff  = 1.8
    L = lambda T: Acoeff + Bcoeff * T
elif olr_type == 'shell_somerville':
    ''' OLR SCHEME FROM SHELL/SOMERVILLE 2004 '''
#     recalculate_moist_adiabats = True
    recalculate_moist_adiabats = False

    if recalculate_moist_adiabats:
        # MetPy's moist_lapse finds the Temp profile by solving an ODE.
        # In order to speed things up, we'll solve it for many different 
        # surface temps and then interpolate.
        nSamples = 200
        minT = 217    #overflow if minT is below 217 ???
        maxT = 350
        pressures = [101117.36, 97634.04827586208, 94150.73655172413, 90667.4248275862, 87184.11310344828, 83700.80137931035, 80217.48965517241, 76734.17793103449, 73250.86620689656, 69767.55448275861, 66284.24275862069, 62800.93103448275, 59317.619310344824, 55834.307586206894, 52350.99586206897, 48867.68413793103, 45384.372413793106, 41901.060689655176, 38417.74896551724, 34934.43724137931, 31451.12551724138, 27967.813793103443, 24484.502068965514, 21001.190344827588, 17517.878620689648, 14034.566896551722, 10551.255172413794, 7067.943448275866, 3584.6317241379274, 101.32000000000001] * units('Pa') # MetPy requires units
        Tsample = np.linspace(minT, maxT, nSamples) * units('K')        # Sample surface temps
        Tdata = np.zeros((nSamples, len(pressures))) * units('K')
        Tdata[:, 0] = Tsample

        print('Calculating moist adiabats...')
        for i in range(nSamples):
            if i%10 == 0: print('{}/{}'.format(i, nSamples))
            Tdata[i, :] = moist_lapse(temperature=Tdata[i, 0], pressure=pressures)
        print('{}/{}'.format(nSamples, nSamples))

        # Keep T constant above 200 hPa for a Tropopause
        for i in range(len(pressures)):
            if pressures[i].magnitude/100 < 200:
                Tdata[:, i] = Tdata[:, i-1]

        # Create the 2d interpolation function: gives function T_moist(p, T_surf)
        interpolated_moist_adiabat_f = interp2d(pressures, Tsample, Tdata)
    
    boundary_layer = 650    #hPa -- from H_middle = 3.6 km
    top_layer      = 500    #hPa -- from H_top    = 5.6 km
#     boundary_layer = 900
#     top_layer = 100
    k = 0.03                #m2/kg
    up_indx = 0
    while pressures[up_indx].magnitude / 100 > top_layer:
        up_indx += 1
    bl_indx = 0
    while pressures[bl_indx].magnitude / 100 > boundary_layer:
        bl_indx += 1
    T_atmos = np.zeros( (lats.shape[0], pressures.shape[0]))

    def L(T):
        # Unfortunately, SciPy's 'interp2d' function returns sorted arrays.
        # This forces us to do a for loop over lats.
        for i in range(len(lats)):
            # We flip the output since, as stated above, it comes out sorted, and we want high pressure first.
            T_atmos[i, :] = np.flip(interpolated_moist_adiabat_f(pressures, T[i]), axis=0)
        esat_bl = humidsat(T_atmos[:, bl_indx], pressures[bl_indx].magnitude / 100)[0]
        optical_depth = 0.622 * k * RH * esat_bl / g
        emis = 0.3 + 0.7 * (1 - np.exp(-optical_depth))
#         print(emis[0])
        return emis * sig * T_atmos[:, up_indx]**4 + (1 - emis) * sig * T**4
    
elif olr_type in ['full_wvf', 'full_no_wvf']:
    ''' FULL BLOWN '''
    if olr_type == 'full_wvf':
        water_vapor_feedback = True
    else:
        water_vapor_feedback = False
        prescribed_vapor = np.load('data/prescirbed_vapor.npz')['arr_0']

#     recalculate_moist_adiabats = True
    recalculate_moist_adiabats = False

    # Use CliMT radiation scheme along with MetPy's moist adiabat calculator
    nLevels = 30
    radiation = climt.RRTMGLongwave(cloud_overlap_method='clear_only')
    state = climt.get_default_state([radiation], x={}, 
                    y={'label' : 'latitude', 'values': lats, 'units' : 'degress N'},
                    mid_levels={'label' : 'mid_levels', 'values': np.arange(nLevels), 'units' : ''},
                    interface_levels={'label' : 'interface_levels', 'values': np.arange(nLevels + 1), 'units' : ''}
                    )

    if recalculate_moist_adiabats:
        # MetPy's moist_lapse finds the Temp profile by solving an ODE.
        # In order to speed things up, we'll solve it for many different 
        # surface temps and then interpolate.
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

        # Create the 2d interpolation function: gives function T_moist(p, T_surf)
        interpolated_moist_adiabat_f = interp2d(pressures, Tsample, Tdata)
    if water_vapor_feedback == False:
        state['specific_humidity'].values[:, :, :] = prescribed_vapor

    def L(T):
        ''' 
        OLR function.
        Outputs OLR given T_surf.
        Assumes moist adiabat structure, uses full blown radiation code from CliMT.
        Sets temp profile with interpolation of moist adiabat calculations from MetPy.
        Sets specific hum profile by assuming constant RH and using humidsat function from Boos
        '''
        # Set surface state
        state['surface_temperature'].values[:] = T
        # Unfortunately, SciPy's 'interp2d' function returns sorted arrays.
        # This forces us to do a for loop over lats.
        for i in range(len(lats)):
            # We flip the output since, as stated above, it comes out sorted, and we want high pressure first.
            state['air_temperature'].values[0, i, :] = np.flip(interpolated_moist_adiabat_f(pressures, T[i]), axis=0)
        # Set specific hum assuming constant RH
        if water_vapor_feedback == True:
            state['specific_humidity'].values[:] = RH_vals * humidsat(state['air_temperature'].values[:], 
                                                            state['air_pressure'].values[:] / 100)[1]
        tendencies, diagnostics = radiation(state)
        return diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=nLevels).values[0]


################################################################################
# SURFACE FLUX
################################################################################
Fs = lambda lat: 0*lat


################################################################################
# NUMERICAL METHODS
################################################################################
# numerical_method = 'euler_for'
# numerical_method = 'euler_back'
numerical_method = 'crank'


################################################################################
# SET UP CRANK-NICOLSON MATRICES
################################################################################
alpha = 0.5

r = D * dt / dy**2

A = np.zeros((len(lats), len(lats)))
B = np.zeros((len(lats), len(lats)))

rng = np.arange(len(lats)-1)

np.fill_diagonal(A, 2 + 2*r)
A[rng, rng+1] = -r   
A[rng+1, rng] = -r   

np.fill_diagonal(B, 2 - 2*r) 
B[rng, rng+1] = r   
B[rng+1, rng] = r   

#insulated boundaries
A[0, 0] = 1; A[0, 1] = -1
A[-1, -2] = 1; A[-1, -1] = -1

B[0, 0] = 1; B[0, 1] = -1
B[-1, -2] = 1; B[-1, -1] = -1

Ainv = np.linalg.inv(A)
C = np.dot(Ainv, B)


################################################################################
### INTEGRATION STEP
################################################################################
def take_step(E, T, alb):
    """
    one step of numerical integration
    """
    if numerical_method == 'euler_for':
        E = E + dt * g/ps * ( (1-alb)*S - L(T) - Fs(lats) ) + dt * D/dy**2 * ( np.roll(E, 1) - 2*E + np.roll(E,-1) )
    elif numerical_method == 'euler_back':
        E = E + dt * g/ps * ( (1-alb)*S - L(T) - Fs(lats) ) + dt * D/dy**2 * ( np.roll(E, 1) - 2*E + np.roll(E,-1) )
        Estar = E + dt * g/ps * ( (1-alb)*S - L(T) - Fs(lats) ) + dt * D/dy**2 * ( np.roll(E, 1) - 2*E + np.roll(E,-1) )
    elif numerical_method == 'crank':
        E = np.dot(C, E) + dt * g/ps * ( (1-alb)*S - L(T) - Fs(lats) )

    E[0] = E[1]
    E[-1] = E[-2]
    T = T_dataset[np.searchsorted(E_dataset, E)]
    if albedo_feedback:
        alb = lats*0 + a2
        alb[np.where(T <= 273.16)] = a1    
    return E, T, alb
