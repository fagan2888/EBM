#!/usr/bin/env python

################################################################################
# This file contains the code of my Diffusive Energy Balance Model.
#   It has customizable OLR, Insolation, Albedo, and more. We use
#   it to test how different processes affect atmospheric 
#   sensitivity.
#
# Henry G. Peterson with Bill Boos -- summer 2018
################################################################################

################################################################################
### IMPORTS
################################################################################
import numpy as np
import climt
from scipy.integrate import quadrature, trapz
from scipy.interpolate import RectBivariateSpline
from time import clock
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from matplotlib import animation, rc
import os

################################################################################
### STYLES
################################################################################
rc('animation', html='html5')
rc('lines', linewidth=2, color='b', markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, 
        linewidth=1.5)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=13)
rc('xtick.major', size=5, width=1.5)
rc('ytick', labelsize=13)
rc('ytick.major', size=5, width=1.5)
rc('legend', fontsize=14)

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

################################################################################
### CLASS 
################################################################################
class Model():
    
    EBM_PATH = os.environ['EBM_PATH']
    
    def __init__(self, dlat=0.5, dtmax_multiple=1.0, max_sim_years=2, tol=0.001):
        self.dlat            = dlat
        self.dlat_rad        = np.deg2rad(dlat)
        # self.dy              = Re * self.dlat_rad
        self.dtmax           = 0.5 * self.dlat_rad**2 / (D / Re**2)
        self.dt              = dtmax_multiple * self.dtmax
        self.max_sim_years   = max_sim_years
        self.secs_in_min     = 60 
        self.secs_in_hour    = 60  * self.secs_in_min 
        self.secs_in_day     = 24  * self.secs_in_hour 
        self.secs_in_year    = 365 * self.secs_in_day 
        self.max_iters       = int(self.max_sim_years * self.secs_in_year / self.dt)
        self.tol             = tol
        self.lats            = np.linspace(-90 + dlat/2, 90 - dlat/2, int(180/dlat) + 1)
        self.lats_rad        = np.deg2rad(self.lats)
        self.lats_bounds     = np.linspace(-90, 90, int(180/dlat) + 2)
        self.cos_lats        = np.cos(np.deg2rad(self.lats))
        self.cos_lats_bounds = np.cos(np.deg2rad(self.lats_bounds))
        self.sin_lats        = np.sin(np.deg2rad(self.lats))
        self.T_dataset       = np.arange(100, 400, 1e-3)
        self.q_dataset       = self._humidsat(self.T_dataset, ps/100)[1]
        self.E_dataset       = cp*self.T_dataset + RH*self.q_dataset*Lv
        self.plot_fluxes     = False
        self.plot_efe        = False


    def _humidsat(self, t, p):
        """
        FROM BOOS:
        % function [esat,qsat,rsat]=_humidsat(t,p)
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


    def initial_temperature(self, initial_condition, low=None, high=None):
        self.initial_condition = initial_condition
        if initial_condition == 'triangle':
            # self.init_temp = high - (high - low) / 90 * np.abs(self.lats)
            self.init_temp = high - (high - low) * np.abs(self.sin_lats)
        elif initial_condition == 'legendre':
            self.init_temp = 2/3*high + 1/3*low - 2/3 * (high-low) * 1/2 * (3 * self.sin_lats**2 - 1)
        elif initial_condition == 'load_data':
            self.init_temp = np.load('T_array.npz')['arr_0'][-1, :]


    def insolation(self, insolation_type, perturb_center=None, 
            perturb_spread=None, perturb_intensity=None):
        self.insolation_type = insolation_type
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
        elif insolation_type == 'perturbation':
            self.perturb_center = perturb_center
            self.perturb_spread = perturb_spread
            self.perturb_intensity = perturb_intensity
            get_S_control = lambda lat: S0*np.cos(np.deg2rad(lat))/np.pi
            func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
            perturb_normalizer, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)
            get_dS = lambda lat: - perturb_intensity/perturb_normalizer * np.exp(-(lat - perturb_center)**2 / (2*perturb_spread**2))
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
            
        self.S = get_S(self.lats)


    def albedo(self, albedo_feedback=False, alb_ice=None, alb_water=None):
        self.albedo_feedback = albedo_feedback
        self.alb_ice = alb_ice
        self.alb_water = alb_water
        if albedo_feedback == True:
            self.init_alb = alb_water * np.ones(len(self.lats))
            self.init_alb[np.where(init_temp <= 273.16)] = alb_ice
        else:
        # From Clark:
        #   self.init_alb = 0.2725 * np.ones(len(lats))
        # Using the below calculation from KiehlTrenberth1997
        # (Reflected Solar - Absorbed Solar) / (Incoming Solar) = (107-67)/342
            self.init_alb = (40 / 342) * np.ones(len(self.lats))

        self.alb = self.init_alb


    def outgoing_longwave(self, olr_type, emissivity=None, A=None, B=None, 
            RH_vert_profile=None, RH_lat_profile=None, gaussian_spread1=None, 
			gaussian_spread2=None, scale_efe=False, constant_spec_hum=False):
        self.olr_type = olr_type
        if olr_type == 'planck':
            ''' PLANCK RADIATION '''
            L = lambda T: emissivity * sig * T**4 
        elif olr_type == 'linear':
            ''' LINEAR FIT '''
            self.A = A
            self.B = B
            L = lambda T: self.A + self.B * T
        # elif olr_type == 'shell_somerville':
        #     ''' OLR SCHEME FROM SHELL/SOMERVILLE 2004 '''
        #     # Create the 2d interpolation function: gives function T_moist(p, T_surf)
        #     moist_data = np.load('data/moist_adiabat_data.npz')
        #     pressures  = moist_data['pressures']
        #     Tsample    = moist_data['Tsample']
        #     Tdata      = moist_data['Tdata']
        #     RH_vert    = moist_data['RH_vert']
        #     interpolated_moist_adiabat_f = interp2d(pressures, Tsample, Tdata)
            
        #     boundary_layer = 650    #hPa -- from H_middle = 3.6 km
        #     top_layer      = 500    #hPa -- from H_top    = 5.6 km
        #     k = 0.03                #m2/kg
        #     up_indx = 0
        #     while pressures[up_indx] / 100 > top_layer:
        #         up_indx += 1
        #     bl_indx = 0
        #     while pressures[bl_indx] / 100 > boundary_layer:
        #         bl_indx += 1
        #     T_atmos = np.zeros( (self.lats.shape[0], pressures.shape[0]))
        
        #     def L(T):
        #         # Unfortunately, SciPy's 'interp2d' function returns sorted arrays.
        #         # This forces us to do a for loop over lats.
        #         for i in range(len(self.lats)):
        #             # We flip the output since, as self.stated above, it comes out sorted, and we want high pressure first.
        #             T_atmos[i, :] = np.flip(interpolated_moist_adiabat_f(pressures, T[i]), axis=0)
        #         esat_bl = _humidsat(T_atmos[:, bl_indx], pressures[bl_indx] / 100)[0]
        #         optical_depth = 0.622 * k * RH * esat_bl / g
        #         emis = 0.3 + 0.7 * (1 - np.exp(-optical_depth))
        #         return emis * sig * T_atmos[:, up_indx]**4 + (1 - emis) * sig * T**4
            
        elif olr_type in ['full_wvf', 'full_no_wvf']:
            ''' FULL BLOWN '''
            if olr_type == 'full_wvf':
                water_vapor_feedback = True
            else:
                water_vapor_feedback = False
        
            # Use CliMT radiation scheme along with MetPy's moist adiabat calculator
            self.nLevels = 30
            radiation = climt.RRTMGLongwave(cloud_overlap_method='clear_only')
            self.state = climt.get_default_state([radiation], x={}, 
                            y={'label' : 'latitude', 'values': self.lats, 'units' : 'degress N'},
                            mid_levels={'label' : 'mid_levels', 'values': np.arange(self.nLevels), 'units' : ''},
                            interface_levels={'label' : 'interface_levels', 'values': np.arange(self.nLevels + 1), 'units' : ''}
                            )
            pressures = self.state['air_pressure'].values[0, 0, :]

            # Vertical RH profile
            self.RH_dist = RH * np.ones( (self.lats.shape[0], self.nLevels) )
            if RH_vert_profile == 'steps':
                for i in range(self.nLevels):
                    # 0-200:    0
                    # 200-300:  0.8
                    # 300-800:  0.2
                    # 800-1000: 0.8
                    if pressures[i]/100 < 200:
                        self.RH_dist[:, i] = 0
                    elif pressures[i]/100 > 300 and pressures[i]/100 < 800:
                        self.RH_dist[:, i] = 0.2
            elif RH_vert_profile == 'zero_top':
                for i in range(self.nLevels):
                    # 0-200:    0
                    # 200-1000: 0.8
                    if pressures[i]/100 < 200:
                        self.RH_dist[:, i] = 0
            else:
                RH_vert_profile = 'constant'
            self.RH_vert_profile = RH_vert_profile

            # Latitudinal RH profile
            gaussian = lambda mu, sigma, lat: np.exp( -(lat - mu)**2 / (2 * sigma**2) )
            if RH_lat_profile == 'gaussian':
                spread = 45
                lat0 = 0
                def shift_dist(RH_dist, lat0):
                    if scale_efe:
                        lat0 *= 0.64
                    RH_dist *=  np.repeat(gaussian(lat0, spread, self.lats), 
                        self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )
                    return RH_dist
                self.RH_dist = shift_dist(self.RH_dist, lat0)
            elif RH_lat_profile == 'mid_level_gaussian':
                spread = gaussian_spread1
                lat0 = 0
                midlevels = np.where( np.logical_and(pressures/100 < 800, pressures/100 > 300) )[0]
                def shift_dist(RH_dist, lat0):
                    if scale_efe:
                        lat0 *= 0.64
                    RH_dist[:, midlevels] =  np.repeat(0.2 + (RH-0.2) * gaussian(lat0, spread, self.lats), 
                        len(midlevels)).reshape( (self.lats.shape[0], len(midlevels)) )
                    return RH_dist
                self.RH_dist = shift_dist(self.RH_dist, lat0)
            elif RH_lat_profile == 'mid_and_upper_level_gaussian':
                spread1 = gaussian_spread1
                spread2 = gaussian_spread2
                lat0 = 0
                midlevels = np.where( np.logical_and(pressures/100 < 800, pressures/100 > 300) )[0]
                upperlevels = np.where( np.logical_and(pressures/100 < 300, pressures/100 > 200) )[0]
                def shift_dist(RH_dist, lat0):
                    if scale_efe:
                        lat0 *= 0.64
                    RH_dist[:, midlevels] =  np.repeat(0.2 + (RH-0.2) * gaussian(lat0, spread1, self.lats), 
                        len(midlevels)).reshape( (self.lats.shape[0], len(midlevels)) )
                    RH_dist[:, upperlevels] =  np.repeat(0.2 + (RH-0.2) * gaussian(lat0, spread2, self.lats), 
                        len(upperlevels)).reshape( (self.lats.shape[0], len(upperlevels)) )
                    return RH_dist
                self.RH_dist = shift_dist(self.RH_dist, lat0)
            else:
                RH_lat_profile = 'constant'
            self.RH_lat_profile = RH_lat_profile
            self.gaussian_spread1 = gaussian_spread1
            self.gaussian_spread2 = gaussian_spread2

            ## Debug:
            # plt.imshow(self.RH_dist.T, extent=(-90, 90, pressures[0]/100, 0), origin='lower', aspect=.1, cmap='BrBG', vmin=0.0, vmax=1.0)
            # plt.colorbar()
            # plt.show()
            # os.sys.exit()

            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load(self.EBM_PATH + '/data/moist_adiabat_data.npz')
            # pressures  = moist_data['pressures']
            Tsample    = moist_data['Tsample']
            Tdata      = moist_data['Tdata']

            # RectBivariateSpline needs increasing x values
            pressures_flipped = np.flip(pressures, axis=0)
            Tdata = np.flip(Tdata, axis=1)
            interpolated_moist_adiabat = RectBivariateSpline(Tsample, pressures_flipped, Tdata)

            if water_vapor_feedback == False:
                T_control = np.load(self.EBM_PATH + '/data/T_array_{}_{}{}_n{}.npz'.format(self.RH_vert_profile, self.RH_lat_profile, self.gaussian_spread1, self.lats.shape[0]))['arr_0']
                T_control = T_control[-1, :]
                Tgrid_control = np.repeat(T_control, 
                        pressures.shape[0]).reshape( (self.lats.shape[0], pressures.shape[0]) )
                air_temp = interpolated_moist_adiabat.ev(Tgrid_control, 
                        self.state['air_pressure'].values[0, :, :])
                self.state['specific_humidity'].values[0, :, :] = self.RH_dist * self._humidsat(air_temp, 
                        self.state['air_pressure'].values[0, :, :] / 100)[1]
                if constant_spec_hum == True:
                    for i in range(self.nLevels):
                        # get area weighted mean of specific_humidity and set all lats on this level to that val
                        qvals = self.state['specific_humidity'].values[0, :, i]
                        q_const = trapz( qvals * 2 * np.pi * Re**2 * self.cos_lats, dx=self.dlat_rad) / (4 * np.pi * Re**2)
                        self.state['specific_humidity'].values[0, :, i] = q_const
       
            def L(T):
                ''' 
                OLR function.
                Outputs OLR given T_surf.
                Assumes moist adiabat structure, uses full blown radiation code from CliMT.
                Sets temp profile with interpolation of moist adiabat calculations from MetPy.
                Sets specific hum profile by assuming constant RH and using _humidsat function from Boos
                '''
                # Set surface state
                self.state['surface_temperature'].values[:] = T
                # Create a 2D array of the T vals and pass to interpolated_moist_adiabat
                #   note: shape of 'air_temperature' is (lons, lats, press) 
                Tgrid = np.repeat(T, pressures.shape[0]).reshape( (self.lats.shape[0], pressures.shape[0]) )
                self.state['air_temperature'].values[0, :, :] = interpolated_moist_adiabat.ev(Tgrid, 
                        self.state['air_pressure'].values[0, :, :])
                # Set specific hum assuming constant RH
                if water_vapor_feedback == True:
                    if RH_lat_profile != 'constant':
                        # Change RH based on ITCZ
                        E = self.E_dataset[np.searchsorted(self.T_dataset, T)]
                        lat0 = self.lats[np.argmax(E)] 
                        self.RH_dist = shift_dist(self.RH_dist, lat0)
                    self.state['specific_humidity'].values[0, :, :] = self.RH_dist * self._humidsat(self.state['air_temperature'].values[0, :, :], 
                            self.state['air_pressure'].values[0, :, :] / 100)[1]
                # CliMT takes over here, this is where the slowdown occurs
                tendencies, diagnostics = radiation(self.state)
                return diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        else:
            os.sys.exit('Invalid keyword for olr_type: {}'.format(self.olr_type))


        self.L = L


    def take_step(self):
        if self.numerical_method == 'explicit':
            self.E = self.E + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T) ) + self.dt * D / Re**2 / self.cos_lats * np.gradient( (np.gradient(self.E, self.dlat_rad) * self.cos_lats), self.dlat_rad)
        elif self.numerical_method == 'implicit':
            self.E = np.dot(self.C, self.E) + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T) )
    
        # insulated boundaries
        self.E[0] = self.E[1]; self.E[-1] = self.E[-2]

        self.T = self.T_dataset[np.searchsorted(self.E_dataset, self.E)]

        if self.albedo_feedback:
            self.alb = self.alb_water * np.ones(len(self.lats))
            self.alb[np.where(T <= 273.16)] = self.alb_ice    


    def _print_progress(self, frame, error):
        ''' Print the progress of the integration '''
        if frame == 0:
            print('frame = {:5d}'.format(0))
        else:
            print('frame = {:5d}; |dT/dt| = {:.2E}'.format(frame, error))

    def solve(self, numerical_method, frames):
        self.numerical_method = numerical_method
        if numerical_method == 'implicit':
            K  = D / Re**2 * self.dt / self.dlat_rad**2
            c1 = K * np.ones(self.lats.shape)
            c2 = K * self.cos_lats_bounds

            J         = self.cos_lats_bounds.size - 1
            weight1   = self.cos_lats_bounds
            weight2   = self.cos_lats
            weightedK = weight1 * K
            Ka1       = weightedK[0:J] / weight2
            Ka3       = weightedK[1:J+1] / weight2
            Ka2       = np.insert(Ka1[1:J], 0, 0) + np.append(Ka3[0:J-1], 0)
            A         = (np.diag(1 + Ka2, k=0) +
                        np.diag(-Ka3[0:J-1], k=1) +
                        np.diag(-Ka1[1:J], k=-1))

            self.C = np.linalg.inv(A)
        #elif numerical_method == 'semi-implicit':
        #    alpha = 0.5
            
        #    r = D * self.dt / self.dy**2
            
        #    A = np.zeros((len(self.lats), len(self.lats)))
        #    B = np.zeros((len(self.lats), len(self.lats)))
            
        #    rng = np.arange(len(self.lats)-1)
            
        #    np.fill_diagonal(A, 2 + 2*r)
        #    A[rng, rng+1] = -r   
        #    A[rng+1, rng] = -r   
            
        #    np.fill_diagonal(B, 2 - 2*r) 
        #    B[rng, rng+1] = r   
        #    B[rng+1, rng] = r   
            
        #    #insulated boundaries
        #    A[0, 0] = 1; A[0, 1] = -1
        #    A[-1, -2] = 1; A[-1, -1] = -1
            
        #    B[0, 0] = 1; B[0, 1] = -1
        #    B[-1, -2] = 1; B[-1, -1] = -1
            
        #    Ainv = np.linalg.inv(A)
        #    self.C = np.dot(Ainv, B)

        print('\nModel Params:')
        print("dtmax:            {:.2f} s / {:.4f} days".format(self.dtmax, self.dtmax / self.secs_in_day))
        print("dt:               {:.2f} s / {:.4f} days = {:.2f} * dtmax".format(self.dt, self.dt / self.secs_in_day, self.dt / self.dtmax))
        print("max_sim_years:    {} years = {:.0f} iterations".format(self.max_sim_years, self.max_iters))
        print("dlat:             {:.2f} degrees".format(self.dlat))
        print("tolerance:        |dT/dt| < {:.2E}".format(self.tol))
        print("frames:           {}\n".format(frames))
        
        print('Insolation Type:   {}'.format(self.insolation_type))
        if self.insolation_type == 'perturbation':
            print('\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}'.format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print('Initial Temp Dist: {}'.format(self.initial_condition))
        print('Albedo Feedback:   {}'.format(self.albedo_feedback))
        print('OLR Scheme:        {}'.format(self.olr_type))
        if self.olr_type == 'linear':
            print('\tA = {:.2f}, B = {:.2f}'.format(self.A, self.B))
        elif self.olr_type in ['full_wvf', 'full_no_wvf']:
            print('\tRH Vertical Profile: {}'.format(self.RH_vert_profile))
            print('\tRH Latitudinal Profile: {}'.format(self.RH_lat_profile))
        print('Numerical Method:  {}\n'.format(self.numerical_method))
        
        T_array   = np.zeros((frames, self.lats.shape[0]))
        E_array   = np.zeros((frames, self.lats.shape[0]))
        alb_array = np.zeros((frames, self.lats.shape[0]))
        L_array   = np.zeros((frames, self.lats.shape[0]))
        if self.olr_type in ['full_wvf', 'full_no_wvf']:
            q_array   = np.zeros((frames, self.lats.shape[0], self.nLevels))
        
        self.T    = self.init_temp
        self.E    = self.E_dataset[np.searchsorted(self.T_dataset, self.T)]
        self.alb  = self.init_alb

        t0          = clock()
        its_per_frame = int(self.max_iters / (frames - 1))
        for frame in range(frames):
            T_array[frame, :]    = self.T
            E_array[frame, :]    = self.E
            L_array[frame, :]    = self.L(self.T)
            alb_array[frame, :]  = self.alb
            if self.olr_type in ['full_wvf', 'full_no_wvf']:
                q_array[frame, :, :] = self.state['specific_humidity'].values[0, :, :]

            error = np.max(np.abs(T_array[frame, :] - T_array[frame-1, :])) / (self.dt * its_per_frame)

            self._print_progress(frame, error)

            if error < self.tol:
                T_array   = T_array[:frame+1, :]
                E_array   = E_array[:frame+1, :]
                alb_array = alb_array[:frame+1, :]
                L_array   = L_array[:frame+1, :]
                if self.olr_type in ['full_wvf', 'full_no_wvf']:
                    q_array      = q_array[:frame+1, :]
                print('Equilibrium reached in {:8.5f} days ({} iterations).'.format(frame * its_per_frame * self.dt / self.secs_in_day, frame * its_per_frame))
                break

            for i in range(its_per_frame):
                # print('{}/{}'.format(i, its_per_frame))
                self.take_step()
        tf = clock()

        if T_array.shape[0] == frames:
            print('Failed to reach equilibrium in {:8.5f} days ({} iterations). |dT/dt| = {:4.16f}'.format(frame * its_per_frame * self.dt / self.secs_in_day, frame * its_per_frame, np.max(np.abs(T_array[-1, :] - T_array[-2, :])) / self.dt))
        print('\nEfficiency: \n{:10.10f} seconds/iteration\n{:10.10f} seconds/sim day\n'.format((tf-t0) / (frame * its_per_frame), (tf-t0) / (frame * its_per_frame) / self.dt * self.secs_in_day))

        self.T_array   = T_array
        self.E_array   = E_array
        self.alb_array = alb_array
        self.L_array   = L_array
        if self.olr_type in ['full_wvf', 'full_no_wvf']:
            self.q_array   = q_array

    def save_data(self):
        # Save data
        np.savez('T_array.npz', self.T_array)
        np.savez('E_array.npz', self.E_array)
        np.savez('L_array.npz', self.L_array)
        np.savez('alb_array.npz', self.alb_array)
        if self.olr_type in ['full_wvf', 'full_no_wvf']:
            np.savez('q_array.npz', self.q_array)


    def _calculate_efe(self):
        """
        EFE = latitude of max of E
        """
        # Get data and final dist
        E_f = self.E_array[-1, :]
        
        # Interp and find roots
        spl = UnivariateSpline(self.lats, E_f, k=4, s=0)
        roots = spl.derivative().roots()
        
        # Find supposed root based on actual data
        max_index = np.argmax(E_f)
        efe_lat = self.lats[max_index]
        
        # Pick up closest calculated root to the supposed one
        min_error_index = np.argmin( np.abs(roots - efe_lat) )
        closest_root = roots[min_error_index]

        self.EFE = closest_root
        self.EFE_rad = np.deg2rad(self.EFE)
        self.ITCZ = 0.64 * self.EFE 
        self.ITCZ_rad = np.deg2rad(self.ITCZ)


    def log_efe(self, fname_efe):
        """
        Write EFE data to a file
        """
        self.plot_efe = True

        print('Calculating EFE...')
        self._calculate_efe()

        with open(fname_efe, 'a') as f:
            if self.insolation_type == 'perturbation':
                data = '{:2d}, {:2.2f}, {:2d}, {:2.16f}, {:2.16f}'.format(self.perturb_center, self.perturb_spread, self.perturb_intensity, self.EFE, self.ITCZ)
            else:
                data = '{:2d}, {:2.2f}, {:2d}, {:2.16f}, {:2.16f}'.format(0, 0, 0, self.EFE, self.ITCZ)
            f.write(data + '\n')
        print('Logged "{}" in "{}"'.format(data, fname_efe))


    def _integrate_lat(self, f, i=-1):
        """
        integrate some array f over phi up to index i
        """
        if i == -1:
            return trapz( f * 2 * np.pi * Re**2 * self.cos_lats, dx=self.dlat_rad ) 
        else:
            return trapz( f[:i+1] * 2 * np.pi * Re**2 * self.cos_lats[:i+1], dx=self.dlat_rad ) 

    # def _calculate_shift(self, fluxes=[]):
        # """
        # Calculate dphi_i for given feedback flux
        # dphi_i = (integral_SLbar + F_i) / -(dF_nf + dF_i)
        # """
        # numerator   = self.integral_SLbar
        # spl         = UnivariateSpline(self.lats_rad, self.flux_no_fb, k=4, s=0)
        # denominator = -spl.derivative()(self.EFE_rad)
        # for flux in fluxes:
        #     spl          = UnivariateSpline(self.lats_rad, flux, k=4, s=0)
        #     # numerator   += spl(self.EFE_rad)
        #     numerator   += spl(0)
        #     denominator -= spl.derivative()(self.EFE_rad)
        # shift = numerator / denominator
        # return np.rad2deg(shift)

    def _calculate_shift(self):
        """
        Calculate dphi using control values
        """
        ctl_data        = np.load('control_data.npz')
        S_ctl           = ctl_data['S']
        L_bar_ctl       = ctl_data['L_bar']
        flux_total_ctl  = 10**-15 * ctl_data['flux_total']
        flux_planck_ctl = 10**-15 * ctl_data['flux_planck']
        flux_wv_ctl     = 10**-15 * ctl_data['flux_wv']
        flux_no_fb_ctl  = 10**-15 * ctl_data['flux_no_fb']

        S = self.S * (1 - self.alb_array[-1, :])
        dS = S - S_ctl

        I_equator = self.lats.shape[0]//2 

        dflux_planck = self.flux_planck - flux_planck_ctl 
        dflux_wv = self.flux_wv - flux_wv_ctl 
        dflux_no_fb = self.flux_no_fb - flux_no_fb_ctl 

        # 1
        numerator = self._integrate_lat(S_ctl - L_bar_ctl, I_equator) + self._integrate_lat(dS, I_equator) + flux_planck_ctl[I_equator] + flux_wv_ctl[I_equator] + dflux_planck[I_equator] + dflux_wv[I_equator]
        denominator = 0

        spl = UnivariateSpline(self.lats_rad, flux_no_fb_ctl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, flux_planck_ctl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, flux_wv_ctl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)

        spl = UnivariateSpline(self.lats_rad, dflux_no_fb, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, dflux_planck, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, dflux_wv, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)

        shift = np.rad2deg(numerator / denominator)
        print(shift)

        # 2
        numerator = flux_total_ctl[I_equator] + self._integrate_lat(dS, I_equator) + dflux_planck[I_equator] + dflux_wv[I_equator]
        denominator = 0 

        spl = UnivariateSpline(self.lats_rad, flux_total_ctl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, dflux_planck, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, dflux_wv, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)

        shift = np.rad2deg(numerator / denominator)
        print(shift)

        # 3
        numerator = self.flux_total[I_equator]
        denominator = 0
        spl = UnivariateSpline(self.lats_rad, self.flux_total, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        shift = np.rad2deg(numerator / denominator)
        print(shift)

        # comparisons
        print(self._integrate_lat(S_ctl - L_bar_ctl, I_equator) + flux_planck_ctl[I_equator] + flux_wv_ctl[I_equator])
        print(flux_total_ctl[I_equator])
        

        # plt.figure(figsize=(16,10))

        # ax1 = plt.subplot(311)
        # ax1.plot(self.sin_lats, flux_planck_ctl)
        # ax1.plot(self.sin_lats, self.flux_planck)
        # ax1.plot(self.sin_lats, dflux_planck)
        # ax1.plot(np.sin(self.EFE_rad), 0, 'ko')
        # ax1.grid()

        # ax2 = plt.subplot(312)
        # ax2.plot(self.sin_lats, flux_wv_ctl)
        # ax2.plot(self.sin_lats, self.flux_wv)
        # ax2.plot(self.sin_lats, dflux_wv)
        # ax2.plot(np.sin(self.EFE_rad), 0, 'ko')
        # ax2.grid()

        # ax3 = plt.subplot(313)
        # ax3.plot(self.sin_lats, flux_no_fb_ctl)
        # ax3.plot(self.sin_lats, self.flux_no_fb)
        # ax3.plot(self.sin_lats, dflux_no_fb)
        # ax3.plot(np.sin(self.EFE_rad), 0, 'ko')
        # ax3.grid()

        # plt.show()

        # os.sys.exit()


    def _calculate_feedback_flux(self, L_flux):
        """
        Perform integral calculation to get F_i = - integral of L - L_i
        (in PW)
        """
        area = self._integrate_lat(1)
        L_flux_bar = 1 / area * self._integrate_lat(L_flux)
        flux = np.zeros( L_flux.shape )
        for i in range( L_flux.shape[0] ):
            flux[i] = -self._integrate_lat(L_flux - L_flux_bar, i)
        return 10**-15 * flux


    def log_feedbacks(self, fname_feedbacks):
        """
        Calculate each feedback flux and log data on the shifts
        """
        print('\nCalculating feedbacks...')

        self.plot_fluxes = True

        E_f = self.E_array[-1, :]
        L_f = self.L_array[-1, :]
        T_f = self.T_array[-1, :]

        # Total
        self.flux_total = 10**-15 * (2 * np.pi * Re * self.cos_lats) * (- ps / g * D / Re) * np.gradient(E_f, self.dlat_rad)

        # All LW Feedbacks
        self.flux_all_fb = self._calculate_feedback_flux(L_f)
        
        # Planck
        emissivity = 0.6
        L_planck = emissivity * sig * T_f**4
        self.flux_planck = self._calculate_feedback_flux(L_planck)
        
        # Water Vapor anom --- use L from other simulation with constant q per vertical level
        L_f = self.L_array[-1, :] 
        L_f_const_q = np.load(self.EBM_PATH + '/data/L_array_constant_q_steps_mid_level_gaussian5_n361.npz')['arr_0'][-1, :]
        self.flux_wv_anom = self._calculate_feedback_flux(L_f - L_f_const_q)

        if self.olr_type == 'full_wvf':
            # Water Vapor --- use L from current T_f and no q
            self.RH_dist[:, :] = 0.0
            L_f_zero_q = self.L(T_f)
            np.savez('L_f_zero_q.npz', L_f_zero_q)
            self.flux_wv = self._calculate_feedback_flux(L_f - L_f_zero_q)

        # No feedbacks
        self.flux_no_fb = self.flux_total - self.flux_all_fb

        if self.insolation_type == 'annual_mean_clark' and self.olr_type == 'full_wvf':
            S_f = self.S * (1 - self.alb_array[-1, :])
            area = self._integrate_lat(1)
            L_bar = 1 / area * self._integrate_lat(L_f)
            np.savez('control_data.npz', S=S_f, L_bar=L_bar, flux_total=self.flux_total,
                flux_planck=self.flux_planck, flux_wv=self.flux_wv, flux_no_fb=self.flux_no_fb)

        self._calculate_shift()
        # I = self.lats.shape[0]//2 + 1
        # integral_SL = trapz( (S_f[:I] - L_f[:I]) * 2 * np.pi * Re**2 * self.cos_lats[:I], dx=self.dlat_rad )
        
        # print('Integral S - L from SP to EQ: {:2.2f} PW/m'.format(10**-15 * integral_SL))
        # print('F(0):                         {:2.2f} PW/m'.format(10**-15 * self.flux_total[I-1]))
        
        # self.integral_SLbar = trapz( (S_f[:I] - L_bar) * 2 * np.pi * Re**2 * self.cos_lats[:I], dx=self.dlat_rad )
        
        # with open(fname_feedbacks, 'a') as f:
        #     f.write('EFE, no_fb, planck, wv, wv_anom, planck+wv, planck+wv_anom\n')

        #     shift_no_fb              = self._calculate_shift([])
        #     shift_planck             = self._calculate_shift([self.flux_planck])
        #     if self.olr_type == 'full_wvf':
        #         shift_wv             = self._calculate_shift([self.flux_wv])
        #     else:
        #         shift_wv             = 0
        #     shift_wv_anom            = self._calculate_shift([self.flux_wv_anom])
        #     if self.olr_type == 'full_wvf':
        #         shift_planck_and_wv  = self._calculate_shift([self.flux_planck, self.flux_wv])
        #     else:
        #         shift_planck_and_wv  = 0
        #     shift_planck_and_wv_anom = self._calculate_shift([self.flux_planck, self.flux_wv_anom])

        #     data = '{:2.2f}, {:2.2f}, {:2.2f}, {:2.2f}, {:2.2f}, {:2.2f}, {:2.2f}'.format(
        #             self.EFE, shift_no_fb, shift_planck, shift_wv, shift_wv_anom,
        #             shift_planck_and_wv, shift_planck_and_wv_anom)
        #     f.write(data + '\n')
        # print('Logged {} into {}'.format(data, fname_feedbacks))


    def save_plots(self):
        """
        Plot various data from the simulation
        """
        ### INITIAL DISTRIBUTIONS
        print('\nPlotting Initial Dists')
        fig, ax = plt.subplots(1, figsize=(16,10))
        
        # radiaiton dist
        SW = self.S * (1 - self.init_alb)
        LW = self.L(self.init_temp)
        ax.plot([-1, 1], [0, 0], 'k--', lw=2)
        ax.plot(self.sin_lats, SW, 'r', lw=2, label='SW (with albedo)', alpha=0.5)
        ax.plot(self.sin_lats, LW, 'g', lw=2, label='LW (init)', alpha=0.5)
        ax.plot(self.sin_lats, SW - LW, 'b', lw=2, label='SW - LW', alpha=1.0)
        
        ax.set_title('SW/LW Radiation (init)')
        ax.set_xlabel('Lat')
        ax.set_ylabel('W/m$^2$')
        ax.legend(loc='upper right')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
        
        plt.tight_layout()
        
        fname = 'init_radiation.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        ### FINAL TEMP DIST
        print('\nPlotting Final T Dist')

        print('Mean T: {:.2f} K'.format(np.mean(self.T_array[-1,:])))

        f, ax = plt.subplots(1, figsize=(16,10))
        ax.plot(self.sin_lats, self.T_array[-1, :], 'k')
        ax.set_title("Final Temperature Distribution")
        ax.set_xlabel('Lat')
        ax.set_ylabel('T (K)')
        ax.grid(c='k', ls='--', lw=1, alpha=0.4)
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
        
        plt.tight_layout()
        
        fname = 'final_temp.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        
        if self.plot_efe:
            ### FIND ITCZ
            print('\nPlotting EFE')
            
            self._calculate_efe()
            E_f = self.E_array[-1, :]
            print("EFE = {:.5f}; ITCZ = {:.5f}".format(self.EFE, self.ITCZ))
            
            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, E_f / 1000, 'c', label='Final Energy Distribution', lw=4)
            min_max = [E_f.min()/1000, E_f.max()/1000]
            ax.plot([np.sin(np.deg2rad(self.EFE)), np.sin(np.deg2rad(self.EFE))], ax.get_ylim(), 'r')
            ax.text(np.sin(np.deg2rad(self.EFE)) + 0.1, np.average(min_max), "EFE $\\approx$ {:.2f}$^\\circ$".format(self.EFE), size=16)
            ax.set_title("Final Energy Distribution")
            ax.legend(fontsize=14, loc="upper left")
            ax.set_xlabel('Lat')
            ax.set_ylabel('E (kJ / kg)')
            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
            
            plt.tight_layout()
            
            fname = 'efe.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()
        
        ### FINAL RADIATION DIST
        print('\nPlotting Final Radiation Dist')

        T_f = self.T_array[-1, :]
        alb_f = self.alb_array[-1, :]
        SW_f = self.S * (1 - alb_f)
        LW_f = self.L_array[-1, :]
        print('Integral of (SW - LW): {:.5f} PW'.format(10**-15 * self._integrate_lat(SW_f - LW_f)))

        f, ax = plt.subplots(1, figsize=(16, 10))
        ax.plot(self.sin_lats, SW_f, 'r', label='$S(1-\\alpha)$')
        ax.plot(self.sin_lats, LW_f, 'b', label='OLR')
        ax.plot(self.sin_lats, SW_f - LW_f, 'g', label='Net')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
        ax.grid()
        ax.legend(loc='upper left')
        ax.set_title("Final Radiation Distributions")
        ax.set_xlabel("Lat")
        ax.set_ylabel("W/m$^2$")
        
        plt.tight_layout()
        
        fname = 'final_radiation.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        if self.plot_fluxes:
            ### FLUXES
            print('\nPlotting Fluxes')

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, self.flux_planck, 'r', label='Planck Feedback')
            ax.plot(self.sin_lats, self.flux_wv_anom, 'b', label='Anomalous Water Vapor Feedback (separate sim)')
            if self.olr_type == 'full_wvf':
                ax.plot(self.sin_lats, self.flux_wv, 'm', label='Total Water Vapor Feedback')
            # ax.plot(self.sin_lats, self.flux_no_fb, 'g', label='Flux Without Feedbacks')
            # ax.plot(self.sin_lats, self.flux_total - self.flux_planck - self.flux_wv, 'g--', label='Flux Without Planck and WV')
            # ax.plot(self.sin_lats, self.flux_all_fb, 'c', label='All LW Feedbacks')
            # ax.plot(self.sin_lats, self.flux_planck + self.flux_wv, 'c--', label='Planck + Total WV')
            ax.plot(self.sin_lats, self.flux_total, 'k', label='Total Energy Flux')
            ax.plot(np.sin(self.EFE_rad), 0,  'ko', label='EFE')
            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
            ax.grid()
            ax.legend()
            ax.set_title("Flux Distributions")
            ax.set_xlabel("Lat")
            ax.set_ylabel("PW")
            
            plt.tight_layout()
            
            fname = 'fluxes.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()
        
        ### LW vs. T
        print('\nPlotting LW vs. T')
        f, ax = plt.subplots(1, figsize=(16, 10))
        ax.set_title("Relationship Between T$_s$ and OLR")
        ax.set_xlabel("T$_s$ (K)")
        ax.set_ylabel("OLR (W/m$^2$)")
        
        func = lambda t, a, b: a + b*t
        
        x_data = self.T_array.flatten()
        y_data = self.L_array.flatten()
        
        popt, pcov = curve_fit(func, x_data, y_data)
        print('A: {:.2f} W/m2, B: {:.2f} W/m2/K'.format(popt[0], popt[1]))
        
        xvals = np.linspace(np.min(x_data), np.max(x_data), 1000)
        yvals = func(xvals, *popt)
        
        ax.plot(x_data, y_data, 'co', ms=2, label='data points: "{}"'.format(self.olr_type))
        ax.plot(xvals, func(xvals, *popt), 'k--', label='linear fit')
        ax.text(np.min(xvals) + 0.1 * (np.max(xvals)-np.min(xvals)), np.mean(yvals), s='A = {:.2f},\nB = {:.2f}'.format(popt[0], popt[1]), size=16)
        ax.legend()
        
        plt.tight_layout()
        
        fname = 'OLR_vs_T_fit_{}.png'.format(self.olr_type)
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        #if self.olr_type in ['full_wvf', 'full_no_wvf']:
        #    ### VERTICAL AIR TEMP
        #    air_temp = self.state['air_temperature'].values[0, :, :]
        #    pressures = self.state['air_pressure'].values[0, 0, :]
        #    for i in range(air_temp.shape[1]):
        #        plt.plot(self.lats, air_temp[:, i], 'k-', lw=1.0)
        #    plt.plot(self.lats, air_temp[:, 0], 'r-')
        #    plt.xlabel('latitude')
        #    plt.ylabel('temp')

        #    plt.figure()
        #    lat = 45
        #    plt.plot(air_temp[int(lat/self.dlat), :], pressures/100, 'k-', label='{}$^\\circ$ lat'.format(lat))
        #    plt.gca().invert_yaxis()
        #    plt.legend()
        #    plt.xlabel('temp')
        #    plt.ylabel('pressure')
        #    
        #    
        #    ### VERTICAL MOISTURE
        #    sp_hum = self.state['air_temperature'].values[0, :, :]
        #    for i in range(sp_hum.shape[1]):
        #        plt.plot(self.lats, sp_hum[:, i], 'k-', lw=1.0)
        #    plt.plot(self.lats, sp_hum[:, 0], 'r-')
        #    plt.xlabel('latitude')
        #    plt.ylabel('sp hum')

        #    plt.figure()
        #    lat = 45
        #    plt.plot(sp_hum[int(lat/self.dlat), :], pressures/100, 'k-', label='{}$^\\circ$ lat'.format(lat))
        #    plt.gca().invert_yaxis()
        #    plt.legend()
        #    plt.xlabel('sp hum')
        #    plt.ylabel('pressure')
