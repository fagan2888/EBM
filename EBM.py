#!/usr/bin/env python

################################################################################
# This file contains the class for a diffusive moist static energy balance 
# model. 
#
# Henry G. Peterson with Bill Boos, 2019
################################################################################

################################################################################
### IMPORTS
################################################################################
import numpy as np
import scipy as sp
import scipy.integrate, scipy.sparse, scipy.optimize, scipy.interpolate
import climt
from time import clock
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import os

################################################################################
### STYLES
################################################################################
rc("animation", html="html5")
rc("lines", linewidth=4, markersize=10)
rc("axes", titlesize=20, labelsize=16, xmargin=0.01, ymargin=0.01, 
        linewidth=1.5)
rc("axes.spines", top=False, right=False)
rc("xtick", labelsize=13)
rc("xtick.major", size=5, width=1.5)
rc("ytick", labelsize=13)
rc("ytick.major", size=5, width=1.5)
rc("legend", fontsize=18)

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
    Diffusive moist static energy balance model.
    """
    
    # This is the path to the folder that contains control sims, moist adiabat data, etc.
    # Either set it as a path in your OS environment or write it here manually.
    EBM_PATH = os.environ["EBM_PATH"]
    # EBM_PATH = "/home/hpeter/ResearchBoos/EBM_files/EBM"
    
    def __init__(self, N_pts=100, dtmax_multiple=1.0, max_sim_years=2, tol=0.001):
        # Setup grid
        self.N_pts = N_pts
        self.dx = 2 / N_pts
        self.sin_lats = np.linspace(-1.0 + self.dx/2, 1.0 - self.dx/2, N_pts)
        self.lats = np.arcsin(self.sin_lats)
        self.latsdeg = np.rad2deg(self.lats)

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
        # alternatively don"t use enhancement factor for non-ideal gas correction
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

        INPUTS
            initial_condition: "triangle"  -> triangle in temp with max at eq
                               "legendre"  -> using first two legendre polys
                               "load_data" -> load T_array.npz data if it is in the folder
            low: lowest temp
            high: highest temp

        OUTPUTS
            Creates array init_temp saved to class.
        """
        self.initial_condition = initial_condition
        if initial_condition == "triangle":
            self.init_temp = high - (high - low) * np.abs(self.sin_lats)
        elif initial_condition == "legendre":
            self.init_temp = 2/3*high + 1/3*low - 2/3 * (high-low) * 1/2 * (3 * self.sin_lats**2 - 1)
        elif initial_condition == "load_data":
            self.init_temp = np.load("T_array.npz")["arr_0"][-1, :]


    def insolation(self, insolation_type, perturb_center=None, perturb_spread=None, perturb_intensity=None):
        """
        Set the incoming shortwave radiation.

        INPUTS
            insolation_type: "perturbation" -> as in Clark et al. 2018 with a gaussian subtracted
            perturb_center: degrees lat -> center of gaussian 
            perturb_spread: degrees lat -> spread of gaussian 
            perturb_intensity: W/m^2 -> M from Clark et al. 2018

        OUTPUTS
            Creates array S saved to class.
        """
        self.insolation_type = insolation_type
        if insolation_type == "perturbation":
            self.perturb_center = perturb_center
            self.perturb_spread = perturb_spread
            self.perturb_intensity = perturb_intensity

            S = S0 / np.pi * np.cos(self.lats)

            func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
            perturb_normalizer, er = sp.integrate.quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)

            self.dS = -perturb_intensity/perturb_normalizer * np.exp(-(self.lats - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))

        self.S = S + self.dS



    def albedo(self, albedo_feedback=False, alb_ice=None, alb_water=None):
        """
        Set surface albedo.

        INPUTS
            albedo_feedback: boolean -> run model with changing albedo or not
            alb_ice: [0, 1] -> albedo of ice
            alb_water: [0, 1] -> albedo of water

        OUTPUTS
            Creates arrays alb, init_alb, and function reset_alb saved to class.
        """
        self.albedo_feedback = albedo_feedback
        self.alb_ice = alb_ice
        self.alb_water = alb_water
        if albedo_feedback == True:
            def reset_alb(T):
                alb = np.ones(self.N_pts)
                alb[:] = self.alb_water
                alb[np.where(T <= 273.16)] = alb_ice
           
                # # Linear functions??
                # ice_buffer = round(0.05 * self.N_pts)

                # i_ice_left = 0
                # i_ice_right = self.N_pts - 1
                # while T[i_ice_left] < 273.16:
                #     i_ice_left += 1
                # while T[i_ice_right] < 273.16:
                #     i_ice_right -= 1

                # if i_ice_left > ice_buffer:
                #     alb[:i_ice_left-ice_buffer] = self.alb_ice
                #     alb[i_ice_left-ice_buffer:i_ice_left+ice_buffer] = self.alb_ice + (self.alb_water - self.alb_ice) / (2 * ice_buffer) * (np.arange(0, 2*ice_buffer, 1))
                # else:
                #     alb[:i_ice_left+ice_buffer] = self.alb_ice + (self.alb_water - self.alb_ice) / (i_ice_left + ice_buffer) * (np.arange(0, i_ice_left + ice_buffer, 1))

                # alb[i_ice_left+ice_buffer:i_ice_right-ice_buffer] = self.alb_water

                # if i_ice_right + ice_buffer < self.N_pts-1:
                #     alb[i_ice_right+ice_buffer:] = self.alb_ice
                #     if i_ice_right-ice_buffer > i_ice_left + ice_buffer:
                #         alb[i_ice_right-ice_buffer:i_ice_right+ice_buffer] = self.alb_water + (self.alb_ice - self.alb_water) / (2 * ice_buffer) * (np.arange(0, 2*ice_buffer, 1))
                #     else:
                #         alb[i_ice_left+ice_buffer:i_ice_right+ice_buffer] = self.alb_water + (self.alb_ice - self.alb_water) / (i_ice_right - i_ice_left) * (np.arange(0, i_ice_right - i_ice_left, 1))
                # else:
                #     if i_ice_right - ice_buffer > i_ice_left + ice_buffer:
                #         alb[i_ice_right-ice_buffer:] = self.alb_water + (self.alb_ice - self.alb_water) / (self.N_pts - i_ice_right + ice_buffer) * (np.arange(0, self.N_pts - i_ice_right + ice_buffer, 1))
                #     else:
                #         alb[i_ice_left+ice_buffer:] = self.alb_water + (self.alb_ice - self.alb_water) / (self.N_pts - i_ice_left - ice_buffer) * (np.arange(0, self.N_pts - i_ice_right + ice_buffer, 1))

                return alb

            self.init_alb = reset_alb(self.init_temp)
            self.reset_alb = reset_alb

            # Debug: Plot init_alb
            # plt.plot(self.sin_lats, self.init_alb)
            # plt.show()
        else:
            # From Clark:
            # self.init_alb = 0.2725 * np.ones(self.N_pts)

            # Using the below calculation from KiehlTrenberth1997
            # (Reflected Solar - Absorbed Solar) / (Incoming Solar) = (107-67)/342 = .11695906432748538011
            # self.init_alb = (40 / 342) * np.ones(self.N_pts)

            # To get an Earth-like T dist (~250 at poles ~300 at EQ)
            self.init_alb = 0.25 * np.ones(self.N_pts)

        if self.albedo_feedback:
            self.ctrl_data = np.load(self.EBM_PATH + "/data/ctrl_alb.npz")
        else:
            self.ctrl_data = np.load(self.EBM_PATH + "/data/ctrl.npz")

        self.alb = self.init_alb


    def outgoing_longwave(self, olr_type, emissivity=None, A=None, B=None):
        """
        Set outgoing longwave radiation.

        INPUTS
            olr_type: "planck" -> L = epsillon * sigma * T^4
                      "linear" -> L = A + B * T 
                      "full_radiation" -> CliMT radiation scheme
                      "full_radiation_2xCO2" -> CliMT radiation scheme with doubled CO2 
                      "full_radiation_no_wv" -> CliMT radiation scheme with prescribed WV
                      "full_radiation_no_lr" -> CliMT radiation scheme with prescribed LR
            emissivity: [0, 1] -> epsillon for "planck" option
            A: float -> A for "linear" option
            B: float -> B for "linear" option

        OUTPUTS
            Creates function L(T) saved within the class.
        """
        self.olr_type = olr_type
        if olr_type == "planck":
            """ PLANCK RADIATION """
            L = lambda T: emissivity * sig * T**4 
        elif olr_type == "linear":
            """ LINEAR FIT """
            self.A = A
            self.B = B
            L = lambda T: self.A + self.B * T
        elif "full_radiation" in olr_type:
            """ FULL BLOWN """
            if olr_type == "full_radiation_no_wv":
                water_vapor_feedback = False
            else:
                water_vapor_feedback = True

            if olr_type == "full_radiation_no_lr":
                lapse_rate_feedback = False
            else:
                lapse_rate_feedback = True
        
            # Use CliMT radiation scheme along with MetPy"s moist adiabat calculator
            self.N_levels = 30   # vertical levels
            self.longwave_radiation = climt.RRTMGLongwave(cloud_overlap_method="clear_only")    # longwave component
            grid = climt.get_grid(nx=1, ny=self.N_pts, nz=self.N_levels)    # setup grid
            grid["latitude"].values[:] = np.rad2deg(self.lats).reshape((self.N_pts, 1))    # force the grid to have my model"s lats
            self.state = climt.get_default_state([self.longwave_radiation], grid_state=grid)    # get the state for this model setup
            pressures = self.state["air_pressure"].values[:, 0, 0]

            if olr_type == "full_radiation_2xCO2":
                # Double CO2
                self.state["mole_fraction_of_carbon_dioxide_in_air"].values[:] = 2 * self.state["mole_fraction_of_carbon_dioxide_in_air"].values[:]

            # Latitudinal RH profile
            gaussian = lambda mu, sigma, lat: np.exp( -(lat - mu)**2 / (2 * sigma**2) )    # quick and dirty gaussian function
            lowerlevels = np.where(pressures/100 > 875)[0]   
            midlevels = np.where(np.logical_and(pressures/100 < 875, pressures/100 > 200))[0]   
            midupperlevels = np.where(np.logical_and(pressures/100 < 200, pressures/100 > 100))[0]   
            def generate_RH_dist(lat_center):
                """
                Make the RH_lat_profile a gaussian and shift its max to the EFE.
                """
                lat_center_deg = np.rad2deg(lat_center)

                RH_dist = np.zeros((self.N_levels, self.N_pts, 1))
                # Set up lower levels as three gaussians
                width_center = 40
                width_left = 90 + lat_center_deg - width_center/2
                width_right = 90 - lat_center_deg - width_center/2

                left = np.where(self.latsdeg < width_left - 90)[0]
                center = np.where(np.logical_and(self.latsdeg > lat_center_deg - width_center/2, self.latsdeg < lat_center_deg + width_center/2))[0]
                right = np.where(self.latsdeg > 90 - width_right)[0]

                spread_center = 1.0*width_center
                spread_left = 1.5*width_left
                spread_right = 1.5*width_right

                RH_dist[lowerlevels, left[0]:left[-1]+1, 0] = np.repeat( 
                    1.0 * gaussian(np.deg2rad(-90), np.deg2rad(spread_left), self.lats[left]), 
                    len(lowerlevels)).reshape( (len(left), len(lowerlevels)) ).T
                RH_dist[lowerlevels, center[0]:center[-1]+1, 0] = np.repeat( 
                    1.0 * gaussian(lat_center, np.deg2rad(spread_center), self.lats[center]), 
                    len(lowerlevels)).reshape( (len(center), len(lowerlevels)) ).T
                RH_dist[lowerlevels, right[0]:right[-1]+1, 0] = np.repeat( 
                    1.0 * gaussian(np.deg2rad(90), np.deg2rad(spread_right), self.lats[right]), 
                    len(lowerlevels)).reshape( (len(right), len(lowerlevels)) ).T

                # Set up mid levels as three gaussians
                width_center = 10
                width_left = 90 + lat_center_deg - width_center/2
                width_right = 90 - lat_center_deg - width_center/2
                
                left = np.where(self.latsdeg < width_left - 90)[0]
                center = np.where(np.logical_and(self.latsdeg > lat_center_deg - width_center/2, self.latsdeg < lat_center_deg + width_center/2))[0]
                right = np.where(self.latsdeg > 90 - width_right)[0]

                spread_center = width_center/4
                spread_left = width_left/2
                spread_right = width_right/2

                RH_dist[midlevels, left[0]:left[-1]+1, 0] = np.repeat( 
                    1.0 * gaussian(np.deg2rad(-90), np.deg2rad(spread_left), self.lats[left]), 
                    len(midlevels)).reshape( (len(left), len(midlevels)) ).T
                RH_dist[midlevels, center[0]:center[-1]+1, 0] = np.repeat( 
                    0.9 * gaussian(lat_center, np.deg2rad(spread_center), self.lats[center]), 
                    len(midlevels)).reshape( (len(center), len(midlevels)) ).T
                RH_dist[midlevels, right[0]:right[-1]+1, 0] = np.repeat( 
                    1.0 * gaussian(np.deg2rad(90), np.deg2rad(spread_right), self.lats[right]), 
                    len(midlevels)).reshape( (len(right), len(midlevels)) ).T
                # # RH Feedback:
                # RH_dist[midlevels, right[0]:right[-1]+1, 0] = np.repeat( 
                #     0.0 * gaussian(np.deg2rad(90), np.deg2rad(spread_right), self.lats[right]), 
                #     len(midlevels)).reshape( (len(right), len(midlevels)) ).T

                # Set up upper levels as one gaussian
                RH_dist[midupperlevels, :, 0] = np.repeat( 
                    0.8 * gaussian(lat_center, np.deg2rad(20), self.lats), 
                    len(midupperlevels)).reshape( (self.N_pts, len(midupperlevels)) ).T
                return RH_dist
            self.generate_RH_dist = generate_RH_dist 
            lat_efe = 0    # the latitude (radians) of the EFE. set this as the center of the gaussian
            self.RH_dist = self.generate_RH_dist(lat_efe)

            # # Debug: Plot RH dist
            # f, ax = plt.subplots(1, figsize=(10,6))
            # levels = np.arange(0, 1.05, 0.05)
            # cf = ax.contourf(self.sin_lats, pressures/100, self.RH_dist[:, :, 0], cmap="BrBG", levels=levels)
            # cb = plt.colorbar(cf, ax=ax, pad=0.1, fraction=0.2)
            # cb.set_ticks(np.arange(0, 1.05, 0.1))
            # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            # ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
            # ax.set_yticks(np.arange(0,1001,100))
            # plt.gca().invert_yaxis()
            # plt.show()

            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load(self.EBM_PATH + "/data/moist_adiabat_data.npz")    # load data from a previous moist adiabat calculation using MetPy
            # pressures  = moist_data["pressures"]
            T_surf_sample = moist_data["T_surf_sample"]    # the surface temp points 
            T_data = moist_data["T_data"]    # the resulting vertical levels temps

            pressures_flipped = np.flip(pressures, axis=0)   # sp.interpolate.RectBivariateSpline needs increasing x values
            T_data = np.flip(T_data, axis=1)
            self.interpolated_moist_adiabat = sp.interpolate.RectBivariateSpline(T_surf_sample, pressures_flipped, T_data)    # this returns an object that has the method .ev() to evaluate the interpolation function

            if water_vapor_feedback == False:
                # prescribe WV from control simulation
                T_control = self.ctrl_data["ctrl_state_temp"][0, :, 0]    # control surface temp
                Tgrid_control = np.repeat(T_control, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                air_temp = self.interpolated_moist_adiabat.ev(Tgrid_control, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(air_temp, self.state["air_pressure"].values[:] / 100)[1]    # q = RH * q_sat

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
                self.state["surface_temperature"].values[:] = T.reshape((self.N_pts, 1))
                if lapse_rate_feedback == False:
                    # Retain LR from control simulations by just shifting all levels by difference at surface
                    Tgrid_diff = np.repeat(T - self.ctrl_data["ctrl_state_temp"][0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
                    self.state["air_temperature"].values[:] = self.ctrl_data["ctrl_state_temp"][:] + Tgrid_diff
                    # # Debug: Show air temps
                    # plt.plot(self.state["air_temperature"].values[:, 2*self.N_pts//3, 0], pressures)
                    # plt.plot(self.ctrl_data["ctrl_state_temp"][:, 2*self.N_pts//3, 0], pressures)
                    # plt.gca().invert_yaxis()
                    # plt.show()
                    # # Debug: Show air temps
                    # for i in range(self.N_levels):
                    #     plt.plot(self.sin_lats, self.state["air_temperature"].values[i, :, 0])
                    # plt.show()
                else:
                    # Create a 2D array of the T vals and pass to self.interpolated_moist_adiabat
                    #   note: shape of "air_temperature" is (lons, lats, press) 
                    Tgrid = np.repeat(T, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                    self.state["air_temperature"].values[:] = self.interpolated_moist_adiabat.ev(Tgrid, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                    # # Debug: Show air temps
                    # plt.plot(self.state["air_temperature"].values[:, 2*self.N_pts//3, 0], pressures)
                    # plt.plot(self.ctrl_data["ctrl_state_temp"][:, 2*self.N_pts//3, 0], pressures)
                    # plt.gca().invert_yaxis()
                    # plt.show()
                    # # Debug: Show air temps
                    # for i in range(self.N_levels):
                    #     plt.plot(self.sin_lats, self.state["air_temperature"].values[i, :, 0])
                    # plt.show()
                if water_vapor_feedback == True:
                    # Shift RH_dist based on ITCZ
                    E = self.E_dataset[np.searchsorted(self.T_dataset, T)]
                    lat_efe = self.lats[np.argmax(E)] 
                    self.RH_dist = self.generate_RH_dist(lat_efe)
                    # Recalculate q
                    self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(self.state["air_temperature"].values[:], self.state["air_pressure"].values[:] / 100)[1]
                    # # Debug: Show q
                    # for i in range(self.N_levels):
                    #     plt.plot(self.sin_lats, self.state["specific_humidity"].values[i, :, 0])
                    # plt.show()

                # CliMT"s FORTRAN code takes over here
                tendencies, diagnostics = self.longwave_radiation(self.state)
                return diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
        else:
            os.sys.exit("Invalid keyword for olr_type: {}".format(self.olr_type))


        self.L = L    # save to class


    def take_step(self):
        """
        Take single time step for integration.

        INPUTS

        OUTPUTS
            Returns E, T, alb arrays for next step.
        """
        # step forward using the take_step_matrix set up in self.solve()
        E_new = self.step_matrix.solve(self.E + self.dt * g / ps * ((1 - self.alb) * self.S - self.L(self.T)))
    
        T_new = self.T_dataset[np.searchsorted(self.E_dataset, E_new)]

        if self.albedo_feedback:
            alb_new = self.reset_alb(T_new)
        else:
            alb_new = self.alb
        
        return E_new, T_new, alb_new


    def solve(self, numerical_method, frames):
        """
        Loop through integration time steps.

        INPUTS 
            numerical_method: "implicit" -> fully implicit method
            frames: int -> number of steps of data to save
        
        OUTPUTS
           Creates set of T, E, alb, and q arrays saved to the class. 
        """
        # begin by computing the take_step_matrix for particular scheme
        self.numerical_method = numerical_method
        
        if numerical_method == "implicit":
            a = D / Re**2 * self.dt / self.dx**2
            sin_lats_plus_half = self.sin_lats + self.dx/2
            sin_lats_minus_half = self.sin_lats - self.dx/2
            data = np.array([1 + 2 * a - a * (sin_lats_plus_half**2 + sin_lats_minus_half**2), 
                            a * (sin_lats_plus_half[:-1]**2 - 1), 
                            a * (sin_lats_minus_half[1:]**2 - 1)])
            diags = np.array([0, 1, -1])
            LHS_matrix = sp.sparse.diags(data, diags)
            # LU factorization (convert to CSC first):
            self.step_matrix = sp.sparse.linalg.splu(LHS_matrix.tocsc())
        else:
            os.sys.exit("Invalid numerical method.")

        # Print some useful information
        print("\nModel Params:")
        print("dtmax:            {:.2f} s / {:.4f} days".format(self.dtmax, self.dtmax / self.secs_in_day))
        print("dt:               {:.2f} s / {:.4f} days = {:.2f} * dtmax".format(self.dt, self.dt / self.secs_in_day, self.dt / self.dtmax))
        print("max_sim_years:    {} years = {:.0f} iterations".format(self.max_sim_years, self.max_iters))
        print("dx:               {:.5f}".format(self.dx))
        print("max dlat:         {:.5f}".format(np.rad2deg(np.max(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("min dlat:         {:.5f}".format(np.rad2deg(np.min(np.abs( (np.roll(self.lats, -1) - self.lats)[:-1])))))
        print("tolerance:        |dT/dt| < {:.2E}".format(self.tol))
        print("frames:           {}".format(frames))
        
        print("\nInsolation Type:   {}".format(self.insolation_type))
        if self.insolation_type == "perturbation":
            print("\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}".format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print("Initial Temp Dist: {}".format(self.initial_condition))
        print("Albedo Feedback:   {}".format(self.albedo_feedback))
        print("OLR Scheme:        {}".format(self.olr_type))
        if self.olr_type == "linear":
            print("\tA = {:.2f}, B = {:.2f}".format(self.A, self.B))
        print("Numerical Method:  {}\n".format(self.numerical_method))
        
        # Setup arrays to save data in
        T_array = np.zeros((frames, self.lats.shape[0]))
        E_array = np.zeros((frames, self.lats.shape[0]))
        alb_array = np.zeros((frames, self.lats.shape[0]))
        L_array = np.zeros((frames, self.lats.shape[0]))
        if "full_radiation" in self.olr_type:
            q_array = np.zeros((frames, self.lats.shape[0], self.N_levels))
        
        self.T = self.init_temp
        self.E = self.E_dataset[np.searchsorted(self.T_dataset, self.T)]
        self.alb = self.init_alb

        # Loop through self.take_step() until converged
        t0 = clock()
        its_per_frame = int(self.max_iters / (frames - 1))
        error = self.tol + 1
        iteration = 0
        frame = 0
        while error > self.tol and frame < frames:
            # # Debug: Plot vertical air temp
            # plt.plot(self.state["air_temperature"].values[:, 200, 0], self.state["air_pressure"].values[:, 200, 0])
            # plt.gca().invert_yaxis()
            # plt.show()
            if iteration % its_per_frame == 0:
                error = np.mean(np.abs(self.T - T_array[frame-1, :]) / (its_per_frame * self.dt))
                T_array[frame, :] = self.T
                E_array[frame, :] = self.E
                L_array[frame, :] = self.L(self.T)
                alb_array[frame, :] = self.alb
                if "full_radiation" in self.olr_type:
                    q_array[frame, :, :] = self.state["specific_humidity"].values[0, :, :]

                # Print progress 
                T_avg = np.mean(self.T)
                self._calculate_efe()
                if frame == 0:
                    print("frame = {:5d}; EFE = {:2.3f}; T_avg = {:3.1f}".format(0, np.rad2deg(self.EFE), T_avg))
                else:
                    print("frame = {:5d}; EFE = {:2.3f}; T_avg = {:3.1f}; |dT/dt| = {:.2E}".format(frame, np.rad2deg(self.EFE), T_avg, error))
                # print("Integral of (SW - LW): {:.5f} PW".format(10**-15 * self._integrate_lat(self.S * (1 - self.alb) - L_array[frame, :])))

                # # Debug: Plot RH dist
                # f, ax = plt.subplots(1, figsize=(10,6))
                # levels = np.arange(0, 1.05, 0.05)
                # cf = ax.contourf(self.sin_lats, self.pressures/100, self.RH_dist[:, :, 0], cmap="BrBG", levels=levels)
                # cb = plt.colorbar(cf, ax=ax, pad=0.1, fraction=0.2)
                # cb.set_ticks(np.arange(0, 1.05, 0.1))
                # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
                # ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
                # ax.set_yticks(np.arange(0,1001,100))
                # plt.gca().invert_yaxis()
                # plt.show()

                frame += 1

            # take a step
            self.E, self.T, self.alb = self.take_step()
            iteration += 1
            if self.tol == 0:
                # never stop if tol = 0
                error = 1

        tf = clock()
        sim_time = tf - t0

        # Truncate arrays
        if iteration-1 % its_per_frame == 0:
            frame += 1
        T_array = T_array[:frame, :]
        E_array = E_array[:frame, :]
        alb_array = alb_array[:frame, :]
        L_array = L_array[:frame, :]
        if "full_radiation" in self.olr_type:
            q_array = q_array[:frame, :]

        # Print exit messages
        print("Equilibrium reached in {:8.5f} days ({} iterations).".format(iteration * self.dt / self.secs_in_day, iteration))

        if frame == frames:
            print("Failed to reach equilibrium in {:8.5f} days ({} iterations). |dT/dt| = {:4.16f}".format(iteration * self.dt / self.secs_in_day, iteration, error))
        
        print("\nEfficiency: \n{:10.10f} seconds/iteration\n{:10.10f} seconds/sim day\n".format(sim_time / iteration, sim_time / (iteration * self.dt / self.secs_in_day)))

        # Save arrays to class
        self.T_array = T_array
        self.E_array = E_array
        self.alb_array = alb_array
        self.L_array = L_array
        if "full_radiation" in self.olr_type:
            self.q_array = q_array

        # Save control simulations
        if self.perturb_intensity == 0 and self.olr_type == "full_radiation":
            flux_total = -self._calculate_feedback_flux(self.S * (1 - self.alb) - self.L(self.T))
            S = self.S*(1 - self.alb)
            L_bar = 1/self._integrate_lat(1) * self._integrate_lat(self.L(self.T))
            ctrl_state_temp = self.state["air_temperature"].values[:]
            if self.albedo_feedback:
                fname = "ctrl_alb.npz"
            else:
                fname = "ctrl.npz"
            np.savez(fname, S=S, L_bar=L_bar, flux_total=flux_total, ctrl_state_temp=ctrl_state_temp)
            print("{} created".format(fname))


    def save_data(self):
        """
        Save arrays of state variables.

        INPUTS

        OUTPUTS
        """
        np.savez("T_array.npz", self.T_array)
        np.savez("E_array.npz", self.E_array)
        np.savez("L_array.npz", self.L_array)
        np.savez("alb_array.npz", self.alb_array)
        if "full_radiation" in self.olr_type:
            np.savez("q_array.npz", self.q_array)


    def _calculate_efe(self):
        """
        EFE = latitude of max of E

        INPUTS

        OUTPUTS
            Creates float EFE saved to class.
        """
        # Interp and find roots
        spl = sp.interpolate.UnivariateSpline(self.lats, self.E, k=4, s=0)
        roots = spl.derivative().roots()
        
        # Find supposed root based on actual data
        max_index = np.argmax(self.E)
        efe_lat = self.lats[max_index]
        
        # Pick up closest calculated root to the supposed one
        min_error_index = np.argmin( np.abs(roots - efe_lat) )
        closest_root = roots[min_error_index]

        self.EFE = closest_root


    def log_efe(self, fname_efe):
        """
        Write EFE data to a file.

        INPUTS
            fname_efe: string -> name of file to save to

        OUTPUTS
        """
        self.plot_efe = True

        print("Calculating EFE...")
        self._calculate_efe()

        with open(fname_efe, "a") as f:
            if self.insolation_type == "perturbation":
                data = "{:2d}, {:2.2f}, {:2d}, {:2.16f}".format(self.perturb_center, self.perturb_spread, self.perturb_intensity, np.rad2deg(self.EFE))
            else:
                data = "{:2d}, {:2.2f}, {:2d}, {:2.16f}".format(0, 0, 0, np.rad2deg(self.EFE))
            f.write(data + "\n")
        print("Logged '{}' in {}".format(data, fname_efe))


    def _integrate_lat(self, f, i=-1):
        """
        Integrate some array f over phi up to index i.

        INPUTS
            f: float, array -> some array or constant to integrate
            i: int -> index in array to integrate up to.

        OUTPUTS
            Returns integral using trapezoidal method.
        """
        if i == -1:
            return np.trapz( f * 2 * np.pi * Re**2 * np.ones(self.N_pts), dx=self.dx ) 
        else:
            if isinstance(f, np.ndarray):
                return np.trapz( f[:i+1] * 2 * np.pi * Re**2, dx=self.dx ) 
            else:
                return np.trapz( f * 2 * np.pi * Re**2, dx=self.dx ) 
                # dx = cos(phi) dphi
                # integral( f * 2 pi * r^2 * cos(phi) dphi )


    def _calculate_shift(self):
        """
        Calculate approximations of shift in ITCZ using corrent simulation and control values.

        INPUTS

        OUTPUTS
        """
        dS = self.S_f - self.S_ctrl
        dL_bar = self.L_bar - self.L_bar_ctrl

        I_equator = self.N_pts//2 

        # Method 1: Do the basic Taylor approx
        numerator = self.flux_total[I_equator]
        denominator = 0

        spl = sp.interpolate.UnivariateSpline(self.lats, self.flux_total, k=4, s=0)
        # denominator -= spl.derivative()(self.EFE)
        denominator -= spl.derivative()(0)

        shift = np.rad2deg(numerator / denominator)
        print("Simple Taylor Shift: {:2.2f}".format(shift))
        print("\tNum / Denom = {}/{}".format(numerator, denominator))

        # Method 2: Use feedbacks relative to control
        numerator = self.flux_total_ctrl[I_equator] + self._integrate_lat(dS - dL_bar, I_equator) + self.delta_flux_planck[I_equator] + self.delta_flux_wv[I_equator] + self.delta_flux_lr[I_equator]
        denominator = 0 

        spl = sp.interpolate.UnivariateSpline(self.lats, self.flux_total_ctrl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, self.delta_flux_planck, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, self.delta_flux_wv, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, self.delta_flux_lr, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)

        shift = np.rad2deg(numerator / denominator)
        print("Shift with Feedbacks: {:2.2f}".format(shift))
        print("\tNum / Denom = {}/{}".format(numerator, denominator))


    def _calculate_feedback_flux(self, L_flux):
        """
        Perform integral calculation to get F_i = - integral of L - L_i 

        INPUTS
            L_flux: array -> OLR due to some flux

        OUTPUTS
            Returns flux array (in PW)
        """
        area = self._integrate_lat(1)
        L_flux_bar = 1 / area * self._integrate_lat(L_flux)
        flux = np.zeros( L_flux.shape )
        for i in range( L_flux.shape[0] ):
            flux[i] = -self._integrate_lat(L_flux - L_flux_bar, i)
        return flux


    def log_feedbacks(self, fname_feedbacks):
        """
        Calculate each feedback flux and log data on the shifts.

        INPUTS
            fname_feedbacks: string -> file to write to.

        OUTPUTS
            Creates arrays for each feedback flux saved to class.
        """
        print("\nCalculating feedbacks...")

        self.plot_fluxes = True

        E_f = self.E_array[-1, :]
        L_f = self.L_array[-1, :]
        T_f = self.T_array[-1, :]

        self.S_f = self.S * (1 - self.alb_array[-1, :])
        area = self._integrate_lat(1)
        self.L_bar = 1 / area * self._integrate_lat(L_f)

        # Get ctrl data
        ctrl_state_temp = self.ctrl_data["ctrl_state_temp"]
        pert_state_temp = np.copy(self.state["air_temperature"].values[:])

        ctrl_state_RH_dist = self.generate_RH_dist(0)
        ctrl_state_qstar = self._humidsat(ctrl_state_temp, self.state["air_pressure"].values[:] / 100)[1]
        ctrl_state_q = ctrl_state_RH_dist * ctrl_state_qstar
        pert_state_RH_dist = self.generate_RH_dist(self.EFE)
        pert_state_q = np.copy(self.state["specific_humidity"].values[:])

        self.S_ctrl = self.ctrl_data["S"]
        self.L_bar_ctrl = self.ctrl_data["L_bar"]

        self.flux_total_ctrl = self.ctrl_data["flux_total"]

        self.T_f_ctrl = ctrl_state_temp[0, :, 0]
        self.ctrl_state_temp = ctrl_state_temp
        self.pert_state_temp = pert_state_temp
        self.ctrl_state_q = ctrl_state_q
        self.pert_state_q = pert_state_q
        
        ## dS
        self.delta_S = -self._calculate_feedback_flux(self.dS)

        ## Total
        E_f_ctrl = self.E_dataset[np.searchsorted(self.T_dataset, self.T_f_ctrl)]
        self.flux_total = -self._calculate_feedback_flux(self.S_f - L_f)
        self.delta_flux_total = self.flux_total - self.flux_total_ctrl

        # All LW Feedbacks
        self.flux_all_fb = self._calculate_feedback_flux(L_f)
        
        ## Planck
        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
        self.state["air_temperature"].values[:] = pert_state_temp - Tgrid_diff
        self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, 0].reshape( (self.N_pts, 1) )
        self.state["specific_humidity"].values[:] = pert_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.L_pert_shifted_T = diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
        self.delta_flux_planck = self._calculate_feedback_flux(L_f - self.L_pert_shifted_T)
        
        ## Water Vapor 
        self.state["air_temperature"].values[:] = pert_state_temp
        self.state["surface_temperature"].values[:] = pert_state_temp[0, :, 0].reshape( (self.N_pts, 1) )
        self.state["specific_humidity"].values[:] = ctrl_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.L_pert_shifted_q =  diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
        self.delta_flux_wv = self._calculate_feedback_flux(L_f - self.L_pert_shifted_q)
        
        ## Lapse Rate
        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
        self.state["air_temperature"].values[:] =  ctrl_state_temp + Tgrid_diff
        self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, 0].reshape( (self.N_pts, 1) )
        self.state["specific_humidity"].values[:] = pert_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.L_pert_shifted_LR = diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
        self.delta_flux_lr = self._calculate_feedback_flux(L_f - self.L_pert_shifted_LR)

        ## Albedo
        self.delta_flux_alb = -self._calculate_feedback_flux(self.S_f - self.dS - self.S_ctrl)

        # No feedbacks
        self.flux_no_fb = self.flux_total - self.flux_all_fb

        self._calculate_shift()

    def predict_efe(self):
        dT = self.T - self.T_f_ctrl
        lambda_pl = -3.15
        lambda_wv = 1.8
        lambda_lr = -0.84

        # # Feedback contributions
        # f, ax = plt.subplots(1, figsize=(16, 10))
        # ax.plot(self.sin_lats, dT, "k")
        # ax.plot(self.sin_lats, lambda_pl * dT, "r")
        # ax.plot(self.sin_lats, lambda_wv * dT, "m")
        # ax.plot(self.sin_lats, lambda_lr * dT, "y")
        # ax.set_xlabel("Lat")
        # ax.set_ylabel("T (K)")
        # ax.grid(c="k", ls="--", lw=1, alpha=0.4)
        # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        # ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
        # plt.tight_layout()
        # plt.show()

        # Predicted Delta Feedback fluxes
        flux_pl = -self._calculate_feedback_flux(lambda_pl * dT)
        flux_wv = -self._calculate_feedback_flux(lambda_wv * dT)
        flux_lr = -self._calculate_feedback_flux(lambda_lr * dT)
        f, ax = plt.subplots(1, figsize=(16, 10))
        ax.plot(self.sin_lats, self.delta_S + flux_pl + flux_wv + flux_lr, "k--")
        ax.plot(self.sin_lats, self.delta_flux_total, "k")
        ax.plot(self.sin_lats, flux_pl, "r--")
        ax.plot(self.sin_lats, self.delta_flux_planck, "r")
        ax.plot(self.sin_lats, flux_wv, "m--")
        ax.plot(self.sin_lats, self.delta_flux_wv, "m")
        ax.plot(self.sin_lats, flux_lr, "y--")
        ax.plot(self.sin_lats, self.delta_flux_lr, "y")
        ax.plot(self.sin_lats, self.delta_S, "c")
        ax.plot(np.sin(self.EFE), 0,  "Xr")
        ax.set_xlabel("Lat")
        ax.set_ylabel("Transport [PW]")
        ax.grid(c="k", ls="--", lw=1, alpha=0.4)
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
        plt.tight_layout()
        plt.show()
        
        # Predicted Flux
        f, ax = plt.subplots(1, figsize=(16, 10))
        ax.plot(self.sin_lats, self.flux_total_ctrl + self.delta_S + flux_pl + flux_wv + flux_lr, "k--")
        ax.plot(self.sin_lats, self.flux_total, "k")
        ax.plot(np.sin(self.EFE), 0,  "Xr")
        ax.set_xlabel("Lat")
        ax.set_ylabel("Transport [PW]")
        ax.grid(c="k", ls="--", lw=1, alpha=0.4)
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
        plt.tight_layout()
        plt.show()

        # Predicted Shift
        dS = self.S_f - self.S_ctrl
        dL_bar = self.L_bar - self.L_bar_ctrl

        I_equator = self.N_pts//2 
        numerator = self.flux_total_ctrl[I_equator] + self._integrate_lat(dS - dL_bar, I_equator) + flux_pl[I_equator] + flux_wv[I_equator] + flux_lr[I_equator]
        denominator = 0 

        spl = sp.interpolate.UnivariateSpline(self.lats, self.flux_total_ctrl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, flux_pl, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, flux_wv, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)
        spl = sp.interpolate.UnivariateSpline(self.lats, flux_lr, k=4, s=0)
        denominator -= spl.derivative()(self.EFE)

        shift = np.rad2deg(numerator / denominator)
        print("Shift with Feedbacks: {:2.2f}".format(shift))
        print("\tNum / Denom = {}/{}".format(numerator, denominator))


    def save_plots(self):
        """
        Plot various data from the simulation

        INPUTS

        OUTPUTS
        """
        ### FINAL TEMP DIST
        print("\nPlotting Final T Dist")
        
        T_avg = np.mean(self.T_array[-1,:])
        print("Mean T: {:.2f} K".format(T_avg))

        f, ax = plt.subplots(1, figsize=(16,10))
        ax.plot(self.sin_lats, self.T_array[-1, :], "k")
        ax.text(0, T_avg, "Mean T = {:.2f} K".format(T_avg), size=16)
        ax.set_title("Final Temperature Distribution")
        ax.set_xlabel("Lat")
        ax.set_ylabel("T (K)")
        ax.grid(c="k", ls="--", lw=1, alpha=0.4)
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
        
        plt.tight_layout()
        
        fname = "final_temp.png"
        plt.savefig(fname, dpi=80)
        print("{} created.".format(fname))
        plt.close()
        
        
        if self.plot_efe:
            ### FIND EFE
            print("\nPlotting EFE")
            
            self._calculate_efe()
            E_f = self.E_array[-1, :]
            print("EFE = {:.5f}".format(np.rad2deg(self.EFE)))
            
            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, E_f / 1000, "c", lw=4)
            ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [0, np.max(E_f)/1000], "r")
            ax.text(np.sin(self.EFE) + 0.1, np.average(E_f)/1000, "EFE $\\approx$ {:.2f}$^\\circ$".format(np.rad2deg(self.EFE)), size=16)
            ax.set_title("Final Energy Distribution")
            ax.set_xlabel("Lat")
            ax.set_ylabel("E (kJ / kg)")
            ax.set_ylim([np.min(E_f)/1000 - 1, np.max(E_f)/1000 + 1])
            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
            
            plt.tight_layout()
            
            fname = "efe.png"
            plt.savefig(fname, dpi=80)
            print("{} created.".format(fname))
            plt.close()
        
        ### RADIATION DISTS
        print("\nPlotting Radiation Dists")

        alb_f = self.alb_array[-1, :]
        SW_f = self.S * (1 - alb_f) 
        SW_i = self.S * (1 - self.init_alb)
        LW_f = self.L_array[-1, :]
        LW_i = self.L(self.init_temp)
        print("Integral of (SW - LW): {:.5f} PW".format(10**-15 * self._integrate_lat(SW_f - LW_f)))

        f, ax = plt.subplots(1, figsize=(16, 10))
        ax.plot(self.sin_lats, SW_f, "r", label="Final $S(1-\\alpha)$")
        ax.plot(self.sin_lats, SW_i, "r--", label="Initial $S(1-\\alpha)$")
        ax.plot(self.sin_lats, LW_f, "b", label="Final OLR")
        ax.plot(self.sin_lats, LW_i, "b--", label="Initial OLR")
        ax.plot(self.sin_lats, SW_f - LW_f, "g", label="Final Net")
        ax.plot(self.sin_lats, SW_i - LW_i, "g--", label="Initial Net")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
        ax.set_ylim([-200, 400])
        ax.set_yticks(np.arange(-200, 401, 50))
        ax.grid()
        ax.legend(loc="upper left")
        ax.set_title("Radiation Distributions")
        ax.set_xlabel("Lat")
        ax.set_ylabel("W/m$^2$")
        
        plt.tight_layout()
        
        fname = "radiation.png"
        plt.savefig(fname, dpi=80)
        print("{} created.".format(fname))
        plt.close()
        
        if self.plot_fluxes:
            ### FLUXES
            print("\nPlotting Fluxes")

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, self.delta_flux_total, "k", label="Total: $F_p - F_{ctrl}$")
            ax.plot(self.sin_lats, self.delta_flux_planck,  "r", label="Planck: $L_p - L$; $T_s$ from $ctrl$")
            ax.plot(self.sin_lats, self.delta_flux_wv, "m", label="WV: $L_p - L$; $q$ from $ctrl$")
            ax.plot(self.sin_lats, self.delta_flux_lr, "y", label="LR: $L_p - L$; $LR$ from $ctrl$")
            ax.plot(self.sin_lats, self.delta_flux_alb, "g", label="AL: $S_p - \\delta S - S_c$")
            ax.plot(self.sin_lats, self.delta_S, "c", label="$\\delta S$")
            ax.plot(self.sin_lats, self.delta_S + self.delta_flux_planck + self.delta_flux_wv + self.delta_flux_lr + self.delta_flux_alb, "k--", label="$\\delta S + \\sum \\delta F_i$")
            ax.plot(np.sin(self.EFE), 0,  "Xr", label="EFE")

            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
            ax.grid()
            ax.legend()
            ax.set_title("Flux Distributions")
            ax.set_xlabel("Lat")
            ax.set_ylabel("PW")
            
            plt.tight_layout()
            
            fname = "fluxes.png"
            plt.savefig(fname, dpi=80)
            print("{} created.".format(fname))
            plt.close()

            ### L'
            print("\nPlotting L Differences")

            L_f = self.L_array[-1, :]

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_T, "r",  label="Planck: $L_p - L$; $T_s$ from $ctrl$")
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_q, "m",  label="WV: $L_p - L$; $q$ from $ctrl$")
            ax.plot(self.sin_lats, L_f - self.L_pert_shifted_LR, "y", label="LR: $L_p - L$; $LR$ from $ctrl$")

            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
            ax.grid()
            ax.legend()
            ax.set_title("$L$ Differences")
            ax.set_xlabel("Lat")
            ax.set_ylabel("W/m$^2$")
            
            plt.tight_layout()
            
            fname = "L_diffs.png"
            plt.savefig(fname, dpi=80)
            print("{} created.".format(fname))
            plt.close()
            
            ### T'
            print("\nPlotting Ts Difference")

            L_f = self.L_array[-1, :]

            f, ax = plt.subplots(1, figsize=(16, 10))
            ax.plot(self.sin_lats, self.T - self.T_f_ctrl, "k",  label="$T_p - T_{ctrl}$")

            ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax.set_xticklabels(["-90", "", "", "-60", "", "", "-30", "", "", "EQ", "", "", "30", "", "", "60", "", "", "90"])
            ax.grid()
            ax.legend()
            ax.set_title("$T$ Difference")
            ax.set_xlabel("Lat")
            ax.set_ylabel("K")
            
            plt.tight_layout()
            
            fname = "Ts_diff.png"
            plt.savefig(fname, dpi=80)
            print("{} created.".format(fname))
            plt.close()

            # ### Vertical T
            # print("\nPlotting Vertical T")

            # lat = 15
            # lat_index = int((lat + 90) / self.dlat)
            # f, ax = plt.subplots(1, figsize=(16, 10))
            # ax.plot(self.pert_state_temp[0, lat_index, :], self.pressures/100, "b",  label="$T_p(z)$")
            # ax.plot(self.ctrl_state_temp[0, lat_index, :],  self.pressures/100, "r",  label="$T_{ctrl}(z)$")

            # ax.grid()
            # ax.legend()
            # ax.set_title("$T(z)$ at 15$^\\circ$ Lat")
            # ax.set_xlabel("K")
            # ax.set_ylabel("hPa")
            # ax.invert_yaxis()
            
            # plt.tight_layout()
            
            # fname = "vertical_T.png"
            # plt.savefig(fname, dpi=80)
            # print("{} created.".format(fname))
            # plt.close()

            # ### Diff in q
            # print("\nPlotting Difference in q")

            # f, ax = plt.subplots(1, figsize=(16, 10))
            
            # im = ax.imshow(self.pert_state_q[:, :].T - self.ctrl_state_q[:, :].T, extent=(-90, 90, 0, 1000), aspect=0.1, cmap="Blues", origin="upper")
            # cb = plt.colorbar(im, ax=ax)
            # cb.set_label("g / g")

            # ax.grid()
            # ax.set_title("Difference in $q$")
            # ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            # ax.set_xlabel("Lat")
            # ax.set_ylabel("hPa")
            # ax.invert_yaxis()
            
            # plt.tight_layout()
            
            # fname = "qdiff.png"
            # plt.savefig(fname, dpi=80)
            # print("{} created.".format(fname))
            # plt.close()


        
        # ### LW vs. T
        # print("\nPlotting LW vs. T")
        # f, ax = plt.subplots(1, figsize=(16, 10))
        # ax.set_title("Relationship Between T$_s$ and OLR")
        # ax.set_xlabel("T$_s$ (K)")
        # ax.set_ylabel("OLR (W/m$^2$)")
        
        # func = lambda t, a, b: a + b*t
        
        # x_data = self.T_array.flatten()
        # y_data = self.L_array.flatten()
        
        # popt, pcov = sp.optimize.curve_fit(func, x_data, y_data)
        # print("A: {:.2f} W/m2, B: {:.2f} W/m2/K".format(popt[0], popt[1]))
        
        # xvals = np.linspace(np.min(x_data), np.max(x_data), 1000)
        # yvals = func(xvals, *popt)
        
        # ax.plot(x_data, y_data, "co", ms=2, label="data points: "{}"".format(self.olr_type))
        # ax.plot(xvals, func(xvals, *popt), "k--", label="linear fit")
        # ax.text(np.min(xvals) + 0.1 * (np.max(xvals)-np.min(xvals)), np.mean(yvals), s="A = {:.2f},\nB = {:.2f}".format(popt[0], popt[1]), size=16)
        # ax.legend()
        
        # plt.tight_layout()
        
        # fname = "OLR_vs_T_fit_{}.png".format(self.olr_type)
        # plt.savefig(fname, dpi=80)
        # print("{} created.".format(fname))
        # plt.close()
        
        #if self.olr_type in ["full_wvf", "full_no_wvf"]:
        #    ### VERTICAL AIR TEMP
        #    air_temp = self.state["air_temperature"].values[0, :, :]
        #    pressures = self.state["air_pressure"].values[0, 0, :]
        #    for i in range(air_temp.shape[1]):
        #        plt.plot(self.lats, air_temp[:, i], "k-", lw=1.0)
        #    plt.plot(self.lats, air_temp[:, 0], "r-")
        #    plt.xlabel("latitude")
        #    plt.ylabel("temp")

        #    plt.figure()
        #    lat = 45
        #    plt.plot(air_temp[int(lat/self.dlat), :], pressures/100, "k-", label="{}$^\\circ$ lat".format(lat))
        #    plt.gca().invert_yaxis()
        #    plt.legend()
        #    plt.xlabel("temp")
        #    plt.ylabel("pressure")
        #    
        #    
        #    ### VERTICAL MOISTURE
        #    sp_hum = self.state["air_temperature"].values[0, :, :]
        #    for i in range(sp_hum.shape[1]):
        #        plt.plot(self.lats, sp_hum[:, i], "k-", lw=1.0)
        #    plt.plot(self.lats, sp_hum[:, 0], "r-")
        #    plt.xlabel("latitude")
        #    plt.ylabel("sp hum")

        #    plt.figure()
        #    lat = 45
        #    plt.plot(sp_hum[int(lat/self.dlat), :], pressures/100, "k-", label="{}$^\\circ$ lat".format(lat))
        #    plt.gca().invert_yaxis()
        #    plt.legend()
        #    plt.xlabel("sp hum")
