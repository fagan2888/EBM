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
from scipy.integrate import quadrature, trapz
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
rc('lines', linewidth=4, markersize=10)
rc('axes', titlesize=20, labelsize=16, xmargin=0.00, ymargin=0.00, linewidth=1.5)
rc('axes.spines', top=False, right=False)
rc('xtick', labelsize=16)
rc('xtick.major', size=5, width=1.5)
rc('ytick', labelsize=16)
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
        self.dx = 2 / N_pts
        self.sin_lats = np.linspace(-1.0 + self.dx/2, 1.0 - self.dx/2, N_pts)
        # self.sin_lats = np.linspace(-1.0 + 1e-3, 1.0 - 1e-3, N_pts)
        # self.dx = self.sin_lats[1] - self.sin_lats[0]
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

        # self.T_basic_state = np.load(self.EBM_PATH + "/data/control_data_N401.npz")["ctl_state_temp"][0, :, 0]
        # self.T_basic_state = 288.6 - 29.3 * (3/2 * self.sin_lats**2 - 1/2) - 4.9 * (35/8 * self.sin_lats**4 - 30/8 * self.sin_lats**2 + 3/8)
        self.T_basic_state = np.polynomial.legendre.Legendre((288.6, 0, -29.3, 0, -4.9))(self.sin_lats)
        # self.T_basic_state = 288.57 - 47.2427 * (3/2 * self.sin_lats**2 - 1/2) + 2.16105 * (35/8 * self.sin_lats**4 - 30/8 * self.sin_lats**2 + 3/8)
        # self.T_basic_state = np.polynomial.legendre.Legendre((288.57, 0, -47.2427, 0, 2.16105))(self.sin_lats)
        # plt.plot(self.sin_lats, self.T_basic_state)
        # plt.show()
        self.E_basic_state = self.E_dataset[np.searchsorted(self.T_dataset, self.T_basic_state)]

        # Boolean options
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

            self.dS = -perturb_intensity/perturb_normalizer * np.exp(-(self.lats - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))
            self.dS *= 40 / 342 # albedo
        elif insolation_type == 'constant':
            self.dS = 4 * np.ones(self.N_pts)
        elif insolation_type == 'basic_state':
            self.dS = np.zeros(self.N_pts)


    def feedback_parameters(self, feedback_parameters_type):
        """
        Set the feedback coefficients
        """
        self.feedback_parameters_type = feedback_parameters_type

        gaussian = lambda mu, sigma, lat: np.exp( -(lat - np.deg2rad(mu))**2 / (2 * np.deg2rad(sigma)**2) )

        if feedback_parameters_type == 'constant':
            # # Soden and Held 2006
            # planck = -3.15 * np.ones(self.N_pts)
            # wv = 1.8 * np.ones(self.N_pts)
            # alb = 0.26 * np.ones(self.N_pts)
            # lr = -0.84 * np.ones(self.N_pts)
            # self.feedback_parameter_sum = planck + wv + alb + lr

            # Roe et al. 2015
            self.feedback_parameter_sum = -2.1 * np.ones(self.N_pts)
        elif feedback_parameters_type == 'subtropics_only':
            # constant = -3.15 + 1.8 + 0.26 - 0.84
            constant = -2.0
            self.feedback_parameter_sum = constant + 3 * gaussian(-15, 8, self.lats) + 3 * gaussian(15, 8, self.lats)
        elif feedback_parameters_type == 'subtropics_and_iceline':
            # constant = -3.15 + 1.8 + 0.26 - 0.84
            constant = -2.0
            self.feedback_parameter_sum = constant + 3 * gaussian(-15, 8, self.lats) + 3 * gaussian(15, 8, self.lats) + 3 * gaussian(-67.5, 2.5, self.lats) + 3 * gaussian(67.5, 2.5, self.lats) 

        f, ax = plt.subplots(1, figsize=(16,10))
        ax.plot(self.sin_lats, self.feedback_parameter_sum, 'k')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        ax.set_ylim([-2.5, 1.5])
        plt.show()


    def take_step(self):
        """
        Take single time step for integration.
        """
        # step forward using the take_step_matrix set up in self.solve()
        dE_new = np.dot(self.take_step_matrix, self.dE + self.dt * g / ps * (self.dS + self.feedback_parameter_sum * self.dT))
    
        # insulated boundaries
        dE_new[0] = dE_new[1]
        dE_new[-1] = dE_new[-2]

        dT_new = self.T_dataset[np.searchsorted(self.E_dataset, self.E_basic_state + dE_new)] - self.T_basic_state

        return dE_new, dT_new


    def _print_progress(self, frame, error):
        """ 
        Print the progress of the integration.
        """
        dT_avg = np.mean(self.dT)
        if frame == 0:
            print('frame = {:5d}; dT_avg = {:3.1f}'.format(0, dT_avg))
        else:
            print('frame = {:5d}; dT_avg = {:3.1f}; |d(dT)/dt| = {:.2E}'.format(frame, dT_avg, error))


    def solve(self, numerical_method, frames):
        """
        Loop through integration time steps.
        """
        # begin by computing the take_step_matrix for particular scheme
        self.numerical_method = numerical_method
        if numerical_method == 'implicit':
            eta = 1
        elif numerical_method == 'explicit':
            eta = 0
        elif numerical_method == 'semi-implicit':
            eta = 0.5
        
        beta = D / Re**2 * self.dt * (1 - self.sin_lats**2) / self.dx**2
        alpha = D / Re**2 * self.dt * self.sin_lats / self.dx

        matrix1 = (np.diag(1 + 2 * eta * beta, k=0) + np.diag(eta * alpha[:-1] - eta * beta[:-1], k=1) + np.diag(-eta * beta[1:] - eta * alpha[1:], k=-1))
        matrix2 = (np.diag(1 - 2 * (1 - eta) * beta, k=0) + np.diag((1 - eta) * beta[:-1] - (1 - eta) * alpha[:-1], k=1) + np.diag((1 - eta) * beta[1:] + (1 - eta) * alpha[1:], k=-1))
         
        self.take_step_matrix = np.dot(np.linalg.inv(matrix1), matrix2)

        # Print some useful information
        print('\nModel Params:')
        print("dtmax:            {:.2f} s / {:.4f} days".format(self.dtmax, self.dtmax / self.secs_in_day))
        print("dt:               {:.2f} s / {:.4f} days = {:.2f} * dtmax".format(self.dt, self.dt / self.secs_in_day, self.dt / self.dtmax))
        print("max_sim_years:    {} years = {:.0f} iterations".format(self.max_sim_years, self.max_iters))
        print("dx:               {:.5f}".format(self.dx))
        print("max dlat:         {:.5f}".format(np.rad2deg(np.max(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("min dlat:         {:.5f}".format(np.rad2deg(np.min(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("tolerance:        |dT/dt| < {:.2E}".format(self.tol))
        print("frames:           {}".format(frames))
        
        print('\nInsolation Type:   {}'.format(self.insolation_type))
        if self.insolation_type == 'perturbation':
            print('\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}'.format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print('Total Feedback Type:   {}'.format(self.feedback_parameters_type))
        print('Numerical Method:  {}\n'.format(self.numerical_method))
        
        # Setup arrays to save data in
        dT_array = np.zeros((frames, self.lats.shape[0]))
        dE_array = np.zeros((frames, self.lats.shape[0]))
        
        self.dT = np.zeros(self.N_pts)
        self.dE = np.zeros(self.N_pts)

        # Loop through self.take_step() until converged
        t0 = clock()
        its_per_frame = int(self.max_iters / (frames - 1))
        error = self.tol + 1
        iteration = 0
        frame = 0
        while error > self.tol and frame < frames:
            # take a step, calculate error 
            dE_new, dT_new = self.take_step()
            error = np.abs(np.mean(dT_new - self.dT) / self.dt)

            self.dE = dE_new
            self.dT = dT_new

            if iteration % its_per_frame == 0:
                dT_array[frame, :] = self.dT
                dE_array[frame, :] = self.dE

                self._print_progress(frame, error)
                frame += 1

            iteration += 1
            if self.tol == 0:
                # never stop if tol = 0
                error = 1

        tf = clock()
        sim_time = tf - t0

        # Truncate arrays
        if iteration-1 % its_per_frame == 0:
            frame += 1
        dT_array = dT_array[:frame, :]
        dE_array = dE_array[:frame, :]

        # Print exit messages
        print('Equilibrium reached in {:8.5f} days ({} iterations).'.format(iteration * self.dt / self.secs_in_day, iteration))

        if frame == frames:
            print('Failed to reach equilibrium in {:8.5f} days ({} iterations). |dT/dt| = {:4.16f}'.format(iteration * self.dt / self.secs_in_day, iteration, error))
        
        print('\nEfficiency: \n{:10.10f} seconds/iteration\n{:10.10f} seconds/sim day\n'.format(sim_time / iteration, sim_time / (iteration * self.dt / self.secs_in_day)))

        # Save arrays to class
        self.dT_array = dT_array
        self.dE_array = dE_array


    def save_data(self):
        """
        Save arrays of state variables.
        """
        np.savez('dT_array.npz', self.dT_array)
        np.savez('dE_array.npz', self.dE_array)


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
        self.ITCZ = 0.64 * self.EFE 


    def log_efe(self, fname_efe):
        """
        Write EFE data to a file
        """
        self.plot_efe = True

        print('Calculating EFE...')
        self._calculate_efe()

        with open(fname_efe, 'a') as f:
            if self.insolation_type == 'perturbation':
                data = '{:2d}, {:2.2f}, {:2d}, {:2.16f}, {:2.16f}'.format(self.perturb_center, self.perturb_spread, self.perturb_intensity, np.rad2deg(self.EFE), np.rad2deg(self.ITCZ))
            else:
                data = '{:2d}, {:2.2f}, {:2d}, {:2.16f}, {:2.16f}'.format(0, 0, 0, np.rad2deg(self.EFE), np.rad2deg(self.ITCZ))
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
                # dx = cos(phi) dphi
                # integral( f * 2 pi * r^2 * cos(phi) dphi )


    def _calculate_shift(self):
        """
        Calculate dphi using control values
        """
        S_ctl           = self.ctl_data['S']
        L_bar_ctl       = self.ctl_data['L_bar']
        flux_total_ctl  = self.ctl_data['flux_total']
        # flux_planck_ctl = self.ctl_data['flux_planck']
        # flux_wv_ctl     = self.ctl_data['flux_wv']
        # flux_no_fb_ctl  = self.ctl_data['flux_no_fb']

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
        print('\nCalculating feedbacks...')

        self.plot_fluxes = True

        E_f = self.E_array[-1, :]
        L_f = self.L_array[-1, :]
        T_f = self.T_array[-1, :]

        self.S_f = self.S * (1 - self.alb_array[-1, :])
        area = self._integrate_lat(1)
        self.L_bar = 1 / area * self._integrate_lat(L_f)

        # # Get CTL data
        # ctl_state_temp = self.ctl_data['ctl_state_temp']
        # pert_state_temp = np.copy(self.state['air_temperature'].values[:, :, :])

        # ctl_state_q = self.RH_dist * self._humidsat(ctl_state_temp[0, :, :], self.state['air_pressure'].values[0, :, :] / 100)[1]
        # pert_state_q = np.copy(self.state['specific_humidity'].values[0, :, :])

        # flux_total_ctl  = self.ctl_data['flux_total']

        # self.T_f_ctl = ctl_state_temp[0, :, 0]
        # self.ctl_state_temp = ctl_state_temp
        # self.pert_state_temp = pert_state_temp
        # self.ctl_state_q = ctl_state_q
        # self.pert_state_q = pert_state_q

        # Total
        # self.flux_total = -(D * ps / g / Re**2) * (2 * np.pi * Re * np.cos(self.lats)) * (np.cos(self.lats) / Re) * np.gradient(E_f, self.dx)
        self.flux_total = -(D * ps / g) * (2 * np.pi * Re * np.cos(self.lats)) * (np.cos(self.lats) / Re) * np.gradient(E_f, self.dx)

        plt.plot(self.sin_lats, self.flux_total)
        plt.show()

        # if self.perturb_intensity == 0 and self.olr_type == 'full_wvf':
        #     np.savez('control_data.npz', S=self.S_f, L_bar=self.L_bar, flux_total=self.flux_total, ctl_state_temp=self.state['air_temperature'].values[:, :, :])
        # self.delta_flux_total = self.flux_total - flux_total_ctl

        # # All LW Feedbacks
        # self.flux_all_fb = self._calculate_feedback_flux(L_f)
        
        # # Planck
        # emissivity = 0.6
        # L_planck = emissivity * sig * T_f**4
        # self.flux_planck1 = self._calculate_feedback_flux(L_planck)

        # Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctl_state_temp[0, :, 0], self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )

        # self.state['air_temperature'].values[0, :, :] =  pert_state_temp - Tgrid_diff
        # self.state['surface_temperature'].values[:] = self.state['air_temperature'].values[0, :, 0]
        # self.state['specific_humidity'].values[0, :, :] = pert_state_q
        # tendencies, diagnostics = self.radiation(self.state)
        # self.L_pert_shifted_T = diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        # self.delta_flux_planck = self._calculate_feedback_flux(L_f - self.L_pert_shifted_T)
        
        # # Water Vapor 
        # RH_copy = self.RH_dist[:, :] 
        # self.RH_dist[:, :] = 0.0
        # L_f_zero_q = self.L(T_f)
        # # np.savez('L_f_zero_q.npz', L_f_zero_q)
        # self.flux_wv1 = self._calculate_feedback_flux(L_f - L_f_zero_q)
        # self.RH_dist[:, :] = RH_copy

        # self.state['air_temperature'].values[0, :, :] = pert_state_temp
        # self.state['surface_temperature'].values[:] = pert_state_temp[0, :, 0]
        # self.state['specific_humidity'].values[0, :, :] = ctl_state_q
        # tendencies, diagnostics = self.radiation(self.state)
        # self.L_pert_shifted_q =  diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        # self.delta_flux_wv = self._calculate_feedback_flux(L_f - self.L_pert_shifted_q)
        
        # # Lapse Rate
        # Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctl_state_temp[0, :, 0], self.nLevels).reshape( (self.lats.shape[0], self.nLevels) )
        # self.state['air_temperature'].values[0, :, :] =  ctl_state_temp + Tgrid_diff
        # self.state['surface_temperature'].values[:] = self.state['air_temperature'].values[0, :, 0]
        # self.state['specific_humidity'].values[0, :, :] = pert_state_q
        # tendencies, diagnostics = self.radiation(self.state)
        # self.L_pert_shifted_LR = diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=self.nLevels).values[0]
        # self.delta_flux_lr = self._calculate_feedback_flux(L_f - self.L_pert_shifted_LR)

        # # No feedbacks
        # self.flux_no_fb = self.flux_total - self.flux_all_fb

        # self._calculate_shift()

    def save_plots(self):
        """
        Plot various data from the simulation
        """
        ### INITIAL DISTRIBUTIONS
        print('\nPlotting Initial Dists')
        fig, ax = plt.subplots(1, figsize=(16,10))
        
        # radiaiton dist
        ax.plot(self.sin_lats, self.dS, 'r', lw=2, label='$R_f$', alpha=0.5)
        ax.plot(self.sin_lats, self.feedback_parameter_sum * self.dT_array[0, :], 'g', lw=2, label='$\sum c_i dT$', alpha=0.5)
        
        ax.set_title('Initial Sources/Sinks')
        ax.set_xlabel('Lat')
        ax.set_ylabel('W/m$^2$')
        ax.legend(loc='upper right')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        
        plt.tight_layout()
        
        fname = 'init_sources_sinks.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()

        ### FEEDBACK PARAMETERS
        print('\nPlotting Feedbacks')
        fig, ax = plt.subplots(1, figsize=(16,10))
        
        ax.plot(self.sin_lats, self.feedback_parameter_sum, 'k', lw=2, label='$\sum c_i$')
        
        ax.set_title('Feedback Parameters')
        ax.set_xlabel('Lat')
        ax.set_ylabel('W/m$^2/K$')
        ax.legend(loc='upper right')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        ax.set_ylim([-2.5, 1.5])
        
        plt.tight_layout()
        
        fname = 'feedback_parameters.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        ### FINAL TEMP DIST
        print('\nPlotting Final dT Dist')

        print('Mean dT: {:.2f} K'.format(np.mean(self.dT_array[-1,:])))

        f, ax = plt.subplots(1, figsize=(16,10))
        ax.plot(self.sin_lats, self.dT_array[-1, :], 'k')
        ax.set_title("Final Temperature Distribution")
        ax.set_xlabel('Lat')
        ax.set_ylabel('$\Delta$T (K)')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        ax.set_ylim([0, 10])
        
        plt.tight_layout()
        
        fname = 'final_temp.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        
        
        ### FINAL MSE DIST
        print('\nPlotting Final dE Dist')

        print('Mean dE: {:.2f} kJ / kg'.format(np.mean(self.dE_array[-1,:])/1000))

        f, ax = plt.subplots(1, figsize=(16,10))
        ax.plot(self.sin_lats, self.dE_array[-1, :] / 1000, 'k')
        ax.set_title("Final MSE Distribution")
        ax.set_xlabel('Lat')
        ax.set_ylabel('$\Delta h$ (kJ / kg)')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        ax.set_ylim([0, 20])
        
        plt.tight_layout()
        
        fname = 'final_mse.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()


        if self.plot_efe:
            ### FIND ITCZ
            print('\nPlotting EFE')
            
            self._calculate_efe()
            E_f = self.E_array[-1, :]
            print("EFE = {:.5f}; ITCZ = {:.5f}".format(np.rad2deg(self.EFE), np.rad2deg(self.ITCZ)))
            
            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, E_f / 1000, 'c', lw=4)
            ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [0, np.max(E_f)/1000], 'r')
            ax.text(np.sin(self.EFE) + 0.1, np.average(E_f)/1000, "EFE $\\approx$ {:.2f}$^\\circ$".format(np.rad2deg(self.EFE)), size=16)
            ax.set_title("Final Energy Distribution")
            ax.set_xlabel('Lat')
            ax.set_ylabel('E (kJ / kg)')
            ax.set_ylim([np.min(E_f)/1000 - 1, np.max(E_f)/1000 + 1])
            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(['-90', '', '', '-60', '', '', '-30', '', '', 'EQ', '', '', '30', '', '', '60', '', '', '90'])
            
            plt.tight_layout()
            
            fname = 'efe.png'
            plt.savefig(fname, dpi=80)
            print('{} created.'.format(fname))
            plt.close()
        
        ### FINAL DISTRIBUTIONS
        print('\nPlotting Final Dists')
        fig, ax = plt.subplots(1, figsize=(16,10))
        
        # radiaiton dist
        ax.plot(self.sin_lats, self.dS, 'r', lw=2, label='$R_f$', alpha=0.5)
        ax.plot(self.sin_lats, self.feedback_parameter_sum * self.dT_array[-1, :], 'g', lw=2, label='$\sum c_i dT$', alpha=0.5)
        
        ax.set_title('Final Sources/Sinks')
        ax.set_xlabel('Lat')
        ax.set_ylabel('W/m$^2$')
        ax.legend(loc='upper right')
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 15))))
        ax.set_xticklabels(np.arange(-90, 91, 15))
        
        plt.tight_layout()
        
        fname = 'final_sources_sinks.png'
        plt.savefig(fname, dpi=80)
        print('{} created.'.format(fname))
        plt.close()
        

