#!/usr/bin/env python

################################################################################
# This file contains the class for a diffusive moist static energy balance 
# model. 
#
# Henry G. Peterson with Bill Boos, 2018
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
rc('legend', fontsize=18)

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
class EnergyBalanceModel():
    """
    Diffusive moist static energy balance model
    """
    
    # This path must be set in your OS environment!
    EBM_PATH = os.environ['EBM_PATH']
    
    def __init__(self, N_pts=100, dtmax_multiple=1.0, max_sim_years=2, tol=0.001):
        # Setup grid
        self.N_pts = N_pts
        self.dx = 2 / (N_pts - 1)
        self.sin_lats = np.linspace(-1.0 + self.dx/2, 1.0 - self.dx/2, int(2/self.dx) + 1)
        self.lats = np.arcsin(self.sin_lats)

        # Calculate stable dt
        diffusivity = D / Re**2 * np.cos(self.lats)**2
        dtmax_diff = 0.5 * self.dx**2 / np.max(diffusivity)
        velocity = D / Re**2 * np.sin(self.lats)
        dtmax_cfl = self.dx / np.max(velocity)
        self.dtmax = np.min([dtmax_diff, dtmax_cfl])
        self.dt = dtmax_multiple * self.dtmax

        # Create useful constants
        self.max_sim_years = max_sim_years
        self.secs_in_min = 60 
        self.secs_in_hour = 60  * self.secs_in_min 
        self.secs_in_day = 24  * self.secs_in_hour 
        self.secs_in_year = 365 * self.secs_in_day 

        # Cut off iterations
        self.max_iters = int(self.max_sim_years * self.secs_in_year / self.dt)

        # Tolerance for dT/dt
        self.tol = tol

        # Datasets for numpy search sorted
        self.T_dataset = np.arange(100, 400, 1e-3)
        self.q_dataset = self._humidsat(self.T_dataset, ps/100)[1]
        self.E_dataset = cp*self.T_dataset + RH*self.q_dataset*Lv

        # Boolean options
        self.plot_fluxes = False
        self.plot_efe = False


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
        """
        Set the initial temperature distribution.
        """
        self.initial_condition = initial_condition
        if initial_condition == 'triangle':
            self.init_temp = high - (high - low) * np.abs(self.sin_lats)
        elif initial_condition == 'legendre':
            self.init_temp = 2/3*high + 1/3*low - 2/3 * (high-low) * 1/2 * (3 * self.sin_lats**2 - 1)
        elif initial_condition == 'load_data':
            self.init_temp = np.load('T_array.npz')['arr_0'][-1, :]


    def insolation(self, insolation_type, perturb_center=None, perturb_spread=None, perturb_intensity=None):
        """
        Set the incoming shortwave radiation.
        """
        self.insolation_type = insolation_type
        if insolation_type == 'perturbation':
            self.perturb_center = perturb_center
            self.perturb_spread = perturb_spread
            self.perturb_intensity = perturb_intensity

            S_control = S0 / np.pi * np.cos(self.lats)

            func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
            perturb_normalizer, er = quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)

            dS = -perturb_intensity/perturb_normalizer * np.exp(-(self.lats - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))
            self.S = S_control + dS


    def albedo(self, albedo_feedback=False, alb_ice=None, alb_water=None):
        """
        Set surface albedo.
        """
        self.albedo_feedback = albedo_feedback
        self.alb_ice = alb_ice
        self.alb_water = alb_water
        if albedo_feedback == True:
            self.init_alb = alb_water * np.ones(len(self.lats))
            self.init_alb[np.where(self.init_temp <= 273.16)] = alb_ice
        else:
        # From Clark:
        #   self.init_alb = 0.2725 * np.ones(len(lats))
        # Using the below calculation from KiehlTrenberth1997
        # (Reflected Solar - Absorbed Solar) / (Incoming Solar) = (107-67)/342 = .11695906432748538011
            self.init_alb = (40 / 342) * np.ones(len(self.lats))

        self.alb = self.init_alb


    def outgoing_longwave(self, olr_type, emissivity=None, A=None, B=None, RH_vert_profile=None, RH_lat_profile=None, gaussian_spread1=None, gaussian_spread2=None, scale_efe=False, constant_spec_hum=False):
        """
        Set outgoing longwave radiation.
        """
        self.olr_type = olr_type
        if olr_type == 'planck':
            """ PLANCK RADIATION """
            L = lambda T: emissivity * sig * T**4 
        elif olr_type == 'linear':
            """ LINEAR FIT """
            self.A = A
            self.B = B
            L = lambda T: self.A + self.B * T
        elif olr_type in ['full_wvf', 'full_no_wvf']:
            """ FULL BLOWN """
            if olr_type == 'full_wvf':
                water_vapor_feedback = True
            else:
                water_vapor_feedback = False
        
            # Use CliMT radiation scheme along with MetPy's moist adiabat calculator
            self.nLevels = 30
            self.radiation = climt.RRTMGLongwave(cloud_overlap_method='clear_only')
            self.state = climt.get_default_state([self.radiation], x={}, 
                            y={'label' : 'latitude', 'values': self.lats, 'units' : 'radians'},
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
            self.RH_vert_profile = RH_vert_profile

            # Latitudinal RH profile
            gaussian = lambda mu, sigma, lat: np.exp( -(np.rad2deg(lat) - mu)**2 / (2 * sigma**2) )
            if RH_lat_profile == 'mid_level_gaussian':
                spread = gaussian_spread1
                lat0 = 0
                midlevels = np.where( np.logical_and(pressures/100 < 800, pressures/100 > 300) )[0]
                def shift_dist(RH_dist, lat0):
                    if scale_efe:
                        lat0 *= 0.64
                    RH_dist[:, midlevels] =  np.repeat(0.2 + (RH-0.2) * gaussian(lat0, spread, self.lats), len(midlevels)).reshape( (self.lats.shape[0], len(midlevels)) )
                    return RH_dist
                self.RH_dist = shift_dist(self.RH_dist, lat0)
            self.RH_lat_profile = RH_lat_profile
            self.gaussian_spread1 = gaussian_spread1
            self.gaussian_spread2 = gaussian_spread2

            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load(self.EBM_PATH + '/data/moist_adiabat_data.npz')
            # pressures  = moist_data['pressures']
            Tsample    = moist_data['Tsample']
            Tdata      = moist_data['Tdata']

            # RectBivariateSpline needs increasing x values
            pressures_flipped = np.flip(pressures, axis=0)
            Tdata = np.flip(Tdata, axis=1)
            self.interpolated_moist_adiabat = RectBivariateSpline(Tsample, pressures_flipped, Tdata)

            if water_vapor_feedback == False:
                T_control = np.load(self.EBM_PATH + '/data/T_array_{}_{}{}_n{}.npz'.format(self.RH_vert_profile, self.RH_lat_profile, self.gaussian_spread1, self.lats.shape[0]))['arr_0']
                T_control = T_control[-1, :]
                Tgrid_control = np.repeat(T_control, 
                        pressures.shape[0]).reshape( (self.lats.shape[0], pressures.shape[0]) )
                air_temp = self.interpolated_moist_adiabat.ev(Tgrid_control, 
                        self.state['air_pressure'].values[0, :, :])
                self.state['specific_humidity'].values[0, :, :] = self.RH_dist * self._humidsat(air_temp, 
                        self.state['air_pressure'].values[0, :, :] / 100)[1]
                if constant_spec_hum == True:
                    for i in range(self.nLevels):
                        # get area weighted mean of specific_humidity and set all lats on this level to that val
                        qvals = self.state['specific_humidity'].values[0, :, i]
                        q_const = trapz( qvals * 2 * np.pi * Re**2 * self.cos_lats, dx=self.dlat_rad) / (4 * np.pi * Re**2)
                        self.state['specific_humidity'].values[0, :, i] = q_const

            self.pressures = pressures
            self.pressures_flipped = pressures_flipped
       
            def L(T):
                """ 
                OLR function.
                Outputs OLR given T_surf.
                Assumes moist adiabat structure, uses full blown radiation code from CliMT.
                Sets temp profile with interpolation of moist adiabat calculations from MetPy.
                Sets specific hum profile by assuming constant RH and using _humidsat function from Boos
                """
                # Set surface state
                self.state['surface_temperature'].values[:] = T
                # Create a 2D array of the T vals and pass to self.interpolated_moist_adiabat
                #   note: shape of 'air_temperature' is (lons, lats, press) 
                Tgrid = np.repeat(T, self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )
                self.state['air_temperature'].values[0, :, :] = self.interpolated_moist_adiabat.ev(Tgrid, 
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
                tendencies, diagnostics = self.radiation(self.state)
                return diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        else:
            os.sys.exit('Invalid keyword for olr_type: {}'.format(self.olr_type))


        self.L = L


    def take_step(self):
        """
        Take single time step for integration.
        """
        if self.numerical_method == 'explicit':
            self.E = self.E + self.dt * g/ps * ((1 - self.alb) * self.S - self.L(self.T)) + self.dt * D / Re**2 * (1 - self.sin_lats**2) / self.dx**2 * (np.roll(self.E, -1) - 2 * self.E + np.roll(self.E, 1)) - self.dt * D / Re**2 * self.sin_lats / self.dx * (np.roll(self.E, -1) - np.roll(self.E, 1)) 
        elif self.numerical_method == 'implicit':
            self.E = np.dot(self.B, self.E)#+ self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T) ))
    
        # insulated boundaries
        self.E[0] = self.E[1]; self.E[-1] = self.E[-2]

        self.T = self.T_dataset[np.searchsorted(self.E_dataset, self.E)]

        if self.albedo_feedback:
            self.alb = self.alb_water * np.ones(len(self.lats))
            self.alb[np.where(self.T <= 273.16)] = self.alb_ice    


    def _print_progress(self, frame, error):
        """ 
        Print the progress of the integration.
        """
        T_avg = np.mean(self.T)
        if frame == 0:
            print('frame = {:5d}; T_avg = {:3.1f}'.format(0, T_avg))
        else:
            print('frame = {:5d}; T_avg = {:3.1f}; |dT/dt| = {:.2E}'.format(frame, T_avg, error))

    def solve(self, numerical_method, frames):
        """
        Loop through integration time steps.
        """
        self.numerical_method = numerical_method

        if numerical_method == 'implicit':
            beta = D / Re**2 * self.dt * (1 - self.sin_lats**2) / self.dx**2
            alpha = D / Re * self.dt * self.sin_lats / self.dx

            A = (np.diag(1 + 2 * beta, k=0) + np.diag(alpha[:-1] - beta[:-1], k=1) + np.diag(-beta[1:] - alpha[1:], k=-1))
            self.B = np.linalg.inv(A)

        print('\nModel Params:')
        print("dtmax:            {:.2f} s / {:.4f} days".format(self.dtmax, self.dtmax / self.secs_in_day))
        print("dt:               {:.2f} s / {:.4f} days = {:.2f} * dtmax".format(self.dt, self.dt / self.secs_in_day, self.dt / self.dtmax))
        print("max_sim_years:    {} years = {:.0f} iterations".format(self.max_sim_years, self.max_iters))
        print("dx:               {:.5f}".format(self.dx))
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
            # plt.plot(self.lats, self.T, 'bo')
            # plt.show()

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
                self.take_step()
        tf = clock()

        if T_array.shape[0] == frames:
            print('Failed to reach equilibrium in {:8.5f} days ({} iterations). |dT/dt| = {:4.16f}'.format(frame * its_per_frame * self.dt / self.secs_in_day, frame * its_per_frame, np.max(np.abs(T_array[-1, :] - T_array[-2, :])) / self.dt))
        if frame > 0:
            print('\nEfficiency: \n{:10.10f} seconds/iteration\n{:10.10f} seconds/sim day\n'.format((tf-t0) / (frame * its_per_frame), (tf-t0) / (frame * its_per_frame) / self.dt * self.secs_in_day))

        self.T_array   = T_array
        self.E_array   = E_array
        self.alb_array = alb_array
        self.L_array   = L_array
        if self.olr_type in ['full_wvf', 'full_no_wvf']:
            self.q_array   = q_array

    def save_data(self):
        """
        Save arrays of state variables.
        """
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
            return trapz( f * 2 * np.pi * Re**2 * np.ones(self.lats.shape[0]), dx=self.dx ) 
        else:
            if isinstance(f, np.ndarray):
                return trapz( f[:i+1] * 2 * np.pi * Re**2, dx=self.dx ) 
            else:
                return trapz( f * 2 * np.pi * Re**2, dx=self.dx ) 


    def _calculate_shift(self):
        """
        Calculate dphi using control values
        """
        if self.albedo_feedback:
            ctl_data = np.load(self.EBM_PATH + '/data/control_data_alb_feedback.npz')
        else:
            ctl_data = np.load(self.EBM_PATH + '/data/control_data.npz')
        S_ctl           = ctl_data['S']
        L_bar_ctl       = ctl_data['L_bar']
        flux_total_ctl  = ctl_data['flux_total']
        # flux_planck_ctl = ctl_data['flux_planck']
        # flux_wv_ctl     = ctl_data['flux_wv']
        # flux_no_fb_ctl  = ctl_data['flux_no_fb']

        dS = self.S_f - S_ctl
        self.delta_S = - self._calculate_feedback_flux(dS)
        dL_bar = self.L_bar - L_bar_ctl

        I_equator = self.lats.shape[0]//2 

        # self.dflux_planck1 = self.flux_planck1 - flux_planck_ctl 
        # self.dflux_wv1 = self.flux_wv1 - flux_wv_ctl 
        # dflux_no_fb = self.flux_no_fb - flux_no_fb_ctl 

        # Method 1: Do the basic Taylor approx
        numerator = self.flux_total[I_equator]
        denominator = 0

        spl = UnivariateSpline(self.lats_rad, self.flux_total, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)

        shift = np.rad2deg(numerator / denominator)
        print('Simple Taylor Shift: {:2.2f}'.format(shift))
        print('\tNum / Denom = {}/{}'.format(numerator, denominator))

        # Method 2: Use feedbacks relative to control
        numerator = flux_total_ctl[I_equator] + 10**-15 * self._integrate_lat(dS - dL_bar, I_equator) + self.delta_flux_planck[I_equator] + self.delta_flux_wv[I_equator] + self.delta_flux_lr[I_equator]
        denominator = 0 

        spl = UnivariateSpline(self.lats_rad, flux_total_ctl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, self.delta_flux_planck, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, self.delta_flux_wv, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)
        spl = UnivariateSpline(self.lats_rad, self.delta_flux_lr, k=4, s=0)
        denominator -= spl.derivative()(self.EFE_rad)

        shift = np.rad2deg(numerator / denominator)
        print('Shift with Feedbacks: {:2.2f}'.format(shift))
        print('\tNum / Denom = {}/{}'.format(numerator, denominator))


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
        if self.insolation_type != 'perturbation':
            print('Calculating EFE...')
            self._calculate_efe()

        print('\nCalculating feedbacks...')

        self.plot_fluxes = True

        E_f = self.E_array[-1, :]
        L_f = self.L_array[-1, :]
        T_f = self.T_array[-1, :]

        self.S_f = self.S * (1 - self.alb_array[-1, :])
        area = self._integrate_lat(1)
        self.L_bar = 1 / area * self._integrate_lat(L_f)

        # Get CTL data
        if self.albedo_feedback:
            ctl_data = np.load(self.EBM_PATH + '/data/control_data_alb_feedback.npz')
        else:
            ctl_data = np.load(self.EBM_PATH + '/data/control_data.npz')

        ctl_state_temp = ctl_data['ctl_state_temp']
        pert_state_temp = np.copy(self.state['air_temperature'].values[:, :, :])

        ctl_state_q = self.RH_dist * self._humidsat(ctl_state_temp[0, :, :], self.state['air_pressure'].values[0, :, :] / 100)[1]
        pert_state_q = np.copy(self.state['specific_humidity'].values[0, :, :])

        flux_total_ctl  = ctl_data['flux_total']

        self.T_f_ctl = ctl_state_temp[0, :, 0]
        self.ctl_state_temp = ctl_state_temp
        self.pert_state_temp = pert_state_temp
        self.ctl_state_q = ctl_state_q
        self.pert_state_q = pert_state_q

        # Total
        self.flux_total = 10**-15 * (2 * np.pi * Re * self.cos_lats) * (- ps / g * D / Re) * np.gradient(E_f, self.dlat_rad)
        self.delta_flux_total = self.flux_total - flux_total_ctl

        # All LW Feedbacks
        self.flux_all_fb = self._calculate_feedback_flux(L_f)
        
        # Planck
        emissivity = 0.6
        L_planck = emissivity * sig * T_f**4
        self.flux_planck1 = self._calculate_feedback_flux(L_planck)

        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctl_state_temp[0, :, 0], self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )

        self.state['air_temperature'].values[0, :, :] =  pert_state_temp - Tgrid_diff
        self.state['surface_temperature'].values[:] = self.state['air_temperature'].values[0, :, 0]
        self.state['specific_humidity'].values[0, :, :] = pert_state_q
        tendencies, diagnostics = self.radiation(self.state)
        self.L_pert_shifted_T = diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        self.delta_flux_planck = self._calculate_feedback_flux(L_f - self.L_pert_shifted_T)
        
        # Water Vapor 
        RH_copy = self.RH_dist[:, :] 
        self.RH_dist[:, :] = 0.0
        L_f_zero_q = self.L(T_f)
        # np.savez('L_f_zero_q.npz', L_f_zero_q)
        self.flux_wv1 = self._calculate_feedback_flux(L_f - L_f_zero_q)
        self.RH_dist[:, :] = RH_copy

        self.state['air_temperature'].values[0, :, :] = pert_state_temp
        self.state['surface_temperature'].values[:] = pert_state_temp[0, :, 0]
        self.state['specific_humidity'].values[0, :, :] = ctl_state_q
        tendencies, diagnostics = self.radiation(self.state)
        self.L_pert_shifted_q =  diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        self.delta_flux_wv = self._calculate_feedback_flux(L_f - self.L_pert_shifted_q)
        
        # Lapse Rate
        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctl_state_temp[0, :, 0], self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )
        self.state['air_temperature'].values[0, :, :] =  ctl_state_temp + Tgrid_diff
        self.state['surface_temperature'].values[:] = self.state['air_temperature'].values[0, :, 0]
        self.state['specific_humidity'].values[0, :, :] = pert_state_q
        tendencies, diagnostics = self.radiation(self.state)
        self.L_pert_shifted_LR = diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        self.delta_flux_lr = self._calculate_feedback_flux(L_f - self.L_pert_shifted_LR)

        # No feedbacks
        self.flux_no_fb = self.flux_total - self.flux_all_fb

        # if self.insolation_type == 'annual_mean_clark' and self.olr_type == 'full_wvf':
        #     np.savez('control_data.npz', S=self.S_f, L_bar=self.L_bar, flux_total=self.flux_total, ctl_state_temp=self.state['air_temperature'].values[:, :, :])

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
            # ax.plot(self.sin_lats, self.flux_total, 'k', label='Total')
            ax.plot(self.sin_lats, self.delta_flux_total, 'k', label='Total: $F_p - F_{ctl}$')
            ax.plot(self.sin_lats, self.delta_flux_planck,  'r', label='Planck: $L_p - L$; $T_s$ from $ctl$')
            ax.plot(self.sin_lats, self.delta_flux_wv, 'm', label='WV: $L_p - L$; $q$ from $ctl$')
            ax.plot(self.sin_lats, self.delta_flux_lr, 'y', label='LR: $L_p - L$; $LR$ from $ctl$')
            ax.plot(self.sin_lats, self.delta_S, 'c', label='$\\delta S$')
            ax.plot(self.sin_lats, self.delta_S + self.delta_flux_planck + self.delta_flux_wv + self.delta_flux_lr, 'k--', label='$\\delta S + \\sum F_i$')
            # ax.plot(self.sin_lats, self.flux_no_fb, 'g', label='Flux Without Feedbacks')
            # ax.plot(self.sin_lats, self.flux_total - self.flux_planck - self.flux_wv, 'g--', label='Flux Without Planck and WV')
            # ax.plot(self.sin_lats, self.flux_all_fb, 'c', label='All LW Feedbacks')
            # ax.plot(self.sin_lats, self.flux_planck + self.flux_wv, 'c--', label='Planck + Total WV')
            # ax.plot(self.sin_lats, self.flux_no_fb + self.flux_all_fb, 'c', label='No FB + All FB')
            # ax.plot(self.sin_lats, self.flux_no_fb + self.flux_wv + self.flux_planck, 'c--', label='No FB + (Planck + Total WV)')
            ax.plot(np.sin(self.EFE_rad), 0,  'Xr', label='EFE')

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

            ### L's
            print('\nPlotting L Differences')

            L_f = self.L_array[-1, :]

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_T, 'r',  label='Planck: $L_p - L$; $T_s$ from $ctl$')
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_q, 'm',  label='WV: $L_p - L$; $q$ from $ctl$')
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_LR, 'y', label='LR: $L_p - L$; $LR$ from $ctl$')

            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
            ax.grid()
            ax.legend()
            ax.set_title("$L$ Differences")
            ax.set_xlabel("Lat")
            ax.set_ylabel("W/m$^2$")
            
            plt.tight_layout()
            
            fname = 'L_diffs.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()
            
            ### T'
            print('\nPlotting Ts Difference')

            L_f = self.L_array[-1, :]

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, T_f - self.T_f_ctl, 'k',  label='$T_p - T_{ctl}$')

            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
            ax.grid()
            ax.legend()
            ax.set_title("$T$ Difference")
            ax.set_xlabel("Lat")
            ax.set_ylabel("K")
            
            plt.tight_layout()
            
            fname = 'Ts_diff.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()

            ### Vertical T
            print('\nPlotting Vertical T')

            lat = 15
            lat_index = int((lat + 90) / self.dlat)
            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.pert_state_temp[0, lat_index, :], self.pressures/100, 'b',  label='$T_p(z)$')
            ax.plot(self.ctl_state_temp[0, lat_index, :],  self.pressures/100, 'r',  label='$T_{ctl}(z)$')

            ax.grid()
            ax.legend()
            ax.set_title("$T(z)$ at 15$^\\circ$ Lat")
            ax.set_xlabel("K")
            ax.set_ylabel("hPa")
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            fname = 'vertical_T.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()

            ### Diff in q
            print('\nPlotting Difference in q')

            f, ax = plt.subplots(1, figsize=(16, 10))
            
            im = ax.imshow(self.pert_state_q[:, :].T - self.ctl_state_q[:, :].T, extent=(-90, 90, 0, 1000), aspect=0.1, cmap='Blues', origin='upper')
            cb = plt.colorbar(im, ax=ax)
            cb.set_label("g / g")

            ax.grid()
            ax.set_title("Difference in $q$")
            ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            ax.set_xlabel("Lat")
            ax.set_ylabel("hPa")
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            fname = 'qdiff.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()


        
        # ### LW vs. T
        # print('\nPlotting LW vs. T')
        # f, ax = plt.subplots(1, figsize=(16, 10))
        # ax.set_title("Relationship Between T$_s$ and OLR")
        # ax.set_xlabel("T$_s$ (K)")
        # ax.set_ylabel("OLR (W/m$^2$)")
        
        # func = lambda t, a, b: a + b*t
        
        # x_data = self.T_array.flatten()
        # y_data = self.L_array.flatten()
        
        # popt, pcov = curve_fit(func, x_data, y_data)
        # print('A: {:.2f} W/m2, B: {:.2f} W/m2/K'.format(popt[0], popt[1]))
        
        # xvals = np.linspace(np.min(x_data), np.max(x_data), 1000)
        # yvals = func(xvals, *popt)
        
        # ax.plot(x_data, y_data, 'co', ms=2, label='data points: "{}"'.format(self.olr_type))
        # ax.plot(xvals, func(xvals, *popt), 'k--', label='linear fit')
        # ax.text(np.min(xvals) + 0.1 * (np.max(xvals)-np.min(xvals)), np.mean(yvals), s='A = {:.2f},\nB = {:.2f}'.format(popt[0], popt[1]), size=16)
        # ax.legend()
        
        # plt.tight_layout()
        
        # fname = 'OLR_vs_T_fit_{}.png'.format(self.olr_type)
        # plt.savefig(fname, dpi=80)
        # print('{} created.'.format(fname))
        # plt.close()
        
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
