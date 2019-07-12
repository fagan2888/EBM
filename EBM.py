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
import sympl
from time import clock
import matplotlib.pyplot as plt
from matplotlib import rc
import os

    
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

# This is the path to the folder that contains control sims, moist adiabat data, etc.
# Either set it as a path in your OS environment or write it here manually.
EBM_PATH = os.environ["EBM_PATH"]
# EBM_PATH = "/home/hpeter/ResearchBoos/EBM_files/EBM"

# Styles:
plt.style.use(EBM_PATH + "/plot_styles.mplstyle")

###########################################################################
### CLASS 
################################################################################
class EnergyBalanceModel():
    """
    Diffusive moist static energy balance model.
    """
    
    def __init__(self, N_pts=401, dtmax_multiple=1e3, max_sim_years=5, tol=1e-8, diffusivity="constant"):
        # Setup grid
        self.N_pts = N_pts
        self.dx = 2 / (N_pts - 1)
        self.sin_lats = np.linspace(-1.0, 1.0, N_pts)
        self.cos_lats = np.sqrt(1 - self.sin_lats**2)
        self.lats = np.arcsin(self.sin_lats)

        # Create useful constants
        self.max_sim_years = max_sim_years
        self.secs_in_min = 60 
        self.secs_in_hour = 60 * self.secs_in_min 
        self.secs_in_day = 24 * self.secs_in_hour 
        self.secs_in_year = 365 * self.secs_in_day 

        # Setup Diffusivity
        self.diffusivity = diffusivity
        if diffusivity == "constant":
            x = np.deg2rad(np.array([-90, 0, 90]))
            y = np.array([D, D, D])
            D_f = sp.interpolate.interp1d(x, y, kind="quadratic")
        elif diffusivity == "cesm2":
            cesm2_data = np.load(EBM_PATH + "/data/D_cesm2.npz")
            sin_lats_cesm2 = cesm2_data["sin_lats"]
            D_cesm2 = cesm2_data["D"]
            # D_cesm2 = cesm2_data["D_filtered"]
            x = np.arcsin(sin_lats_cesm2)
            y = Re**2 / ps * g * D_cesm2
            D_f = sp.interpolate.interp1d(x, y, kind="quadratic")
            # NOTE: D_bar = 2.54523E-04
        elif diffusivity == "D1":
            def D_f(lats):
                # D_avg = 2.54523e-4
                D_avg = ps / g * D / Re**2
                lat0 = 15
                L_trop = 2 * np.sin(np.deg2rad(lat0))
                L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
                D_trop = 4.5e-4
                D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
                Diff =  D_trop * np.ones(lats.shape)
                Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
                return g/ps*Re**2 * Diff
        elif diffusivity == "D2":
            def D_f(lats):
                # D_avg = 2.54523e-4
                D_avg = ps / g * D / Re**2
                lat0 = 15
                L_trop = 2 * np.sin(np.deg2rad(lat0))
                L_extrop = 2 * (1 - np.sin(np.deg2rad(lat0)))
                D_trop = 0.5e-4
                D_extrop = (2*D_avg - D_trop*L_trop)/L_extrop
                Diff =  D_trop * np.ones(lats.shape)
                Diff[np.where(np.logical_or(np.rad2deg(lats) <= -lat0, np.rad2deg(lats) > lat0))] = D_extrop
                return g/ps*Re**2 * Diff

        self.D = D_f(self.lats)
        # print("D_bar = {:1.5E}".format(ps/g/Re**2 * 1/self._integrate_lat(1) * self._integrate_lat(self.D)))

        self.sin_lats_mids = (self.sin_lats - self.dx/2)[1:]
        self.lats_mids = np.arcsin(self.sin_lats_mids)
        self.D_mids = D_f(self.lats_mids)
        
        # # Debug: Plot D
        # f, ax = plt.subplots(1)
        # ax.plot(self.sin_lats, ps / g * self.D / Re**2)
        # ax.set_ylim([0, 0.0005])
        # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
        # plt.show()

        # Calculate stable dt
        diffusivity = self.D / Re**2 * np.cos(self.lats)**2
        dtmax_diff = 0.5 * self.dx**2 / np.max(diffusivity)
        velocity = self.D / Re**2 * np.sin(self.lats)
        dtmax_cfl = self.dx / np.max(velocity)
        self.dtmax = np.min([dtmax_diff, dtmax_cfl])
        self.dt = dtmax_multiple * self.dtmax

        # Cut off iterations
        self.max_iters = int(self.max_sim_years * self.secs_in_year / self.dt)

        # Tolerance for dT/dt
        self.tol = tol

        # Datasets for numpy search sorted
        self.T_dataset = np.arange(33, 350, 1e-3)
        self.q_dataset = self._humidsat(self.T_dataset, ps/100)[1]
        self.E_dataset = cp*self.T_dataset + RH*self.q_dataset*Lv

        # Boolean options
        self.plot_transports = False

        # Control data
        self.ctrl_data = np.load(EBM_PATH + "/data/ctrl.npz")


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
            self.init_temp = np.load("simulation_data.npz")["T"][-1, :]


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

            S_basic = S0 / np.pi * self.cos_lats

            func = lambda y: 0.5 * np.exp(-(y - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2)) * np.cos(y)
            perturb_normalizer, er = sp.integrate.quadrature(func, -np.pi/2, np.pi/2, tol=1e-16, rtol=1e-16, maxiter=1000)

            self.dS = -perturb_intensity/perturb_normalizer * np.exp(-(self.lats - np.deg2rad(perturb_center))**2 / (2*np.deg2rad(perturb_spread)**2))

            # print(10**-15*self._integrate_lat(perturb_intensity))

        self.S = S_basic + self.dS


    def albedo(self, al_feedback=False, alb_ice=None, alb_water=None):
        """
        Set surface albedo.

        INPUTS
            al_feedback: boolean -> run model with changing albedo or not
            alb_ice: [0, 1] -> albedo of ice
            alb_water: [0, 1] -> albedo of water

        OUTPUTS
            Creates arrays alb, init_alb, and function reset_alb saved to class.
        """
        self.al_feedback = al_feedback
        self.alb_ice = alb_ice
        self.alb_water = alb_water
        if al_feedback == True:
            def reset_alb(T):
                alb = np.ones(self.N_pts)
                alb[:] = self.alb_water
                alb[np.where(T <= 273.16 - 10)] = alb_ice
                return alb

            self.init_alb = reset_alb(self.init_temp)
            self.reset_alb = reset_alb
        else:
            self.init_alb = self.ctrl_data["alb"]

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
            if "no_wv" in olr_type:
                wv_feedback = False
            else:
                wv_feedback = True

            if "no_lr" in olr_type:
                lr_feedback = False
            else:
                lr_feedback = True

            if "_rh" in olr_type:
                rh_feedback = True
            else:
                rh_feedback = False

            if "homog" in olr_type:
                homog_olr = True
            else:
                homog_olr = False

            # Use CliMT radiation scheme along with MetPy"s moist adiabat calculator
            self.N_levels = 30   # vertical levels
            self.longwave_radiation = climt.RRTMGLongwave(cloud_overlap_method="clear_only")    # longwave component
            grid = climt.get_grid(nx=1, ny=self.N_pts, nz=self.N_levels)    # setup grid
            grid["latitude"].values[:] = np.rad2deg(self.lats).reshape((self.N_pts, 1))    # force the grid to have my model"s lats
            self.state = climt.get_default_state([self.longwave_radiation], grid_state=grid)    # get the state for this model setup
            pressures = self.state["air_pressure"].values[:, 0, 0]

            if "2xCO2" in olr_type:
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
                RH_dist = np.zeros((self.N_levels, self.N_pts, 1))

                # Set up lower levels as constant
                RH_dist[lowerlevels, :, 0] = 0.9 

                # Set up mid levels as four gaussians
                width_center = np.sin(np.deg2rad(30))
                width_left = 1 + np.sin(lat_center) - width_center/2
                width_right = 1 - np.sin(lat_center) - width_center/2
                
                left = np.where(self.sin_lats <= width_left - 1)[0]
                centerL = np.where(np.logical_and(self.sin_lats > np.sin(lat_center) - width_center/2, self.sin_lats <= np.sin(lat_center)))[0]
                centerR = np.where(np.logical_and(self.sin_lats > np.sin(lat_center), self.sin_lats < np.sin(lat_center) + width_center/2))[0]
                right = np.where(self.sin_lats >= 1 - width_right)[0]

                spread_left = 1/4*width_left
                spread_centerL = 1/8*width_center
                spread_centerR = 1/8*width_center
                spread_right = 1/4*width_right

                ## RH Feedback:
                RH_min_ctrl = 0.145
                if rh_feedback:
                    slope_L = -0.0203954292274862
                    slope_R = 0.013515143699796586
                else:
                    slope_L = 0
                    slope_R = 0

                RH_L_max = 0.9
                RH_R_max = 0.9
                RH_C_max = 0.8
                RH_L_min = RH_min_ctrl + slope_L*np.rad2deg(lat_center)
                RH_R_min = np.max([RH_min_ctrl + slope_R*np.rad2deg(lat_center), 0])

                if width_left > 0:
                    RH_dist[midlevels, left[0]:left[-1]+1, 0] = np.repeat( 
                        RH_L_min + (RH_L_max - RH_L_min) * gaussian(-1, spread_left, self.sin_lats[left]), 
                        len(midlevels)).reshape( (len(left), len(midlevels)) ).T
                RH_dist[midlevels, centerL[0]:centerL[-1]+1, 0] = np.repeat( 
                    RH_L_min + (RH_C_max - RH_L_min) * gaussian(np.sin(lat_center), spread_centerL, self.sin_lats[centerL]), 
                    len(midlevels)).reshape( (len(centerL), len(midlevels)) ).T
                RH_dist[midlevels, centerR[0]:centerR[-1]+1, 0] = np.repeat( 
                    RH_R_min + (RH_C_max - RH_R_min) * gaussian(np.sin(lat_center), spread_centerR, self.sin_lats[centerR]), 
                    len(midlevels)).reshape( (len(centerR), len(midlevels)) ).T
                RH_dist[midlevels, right[0]:right[-1]+1, 0] = np.repeat( 
                    RH_R_min + (RH_R_max - RH_R_min) * gaussian(1, spread_right, self.sin_lats[right]), 
                    len(midlevels)).reshape( (len(right), len(midlevels)) ).T

                # Set up upper levels as one gaussian
                RH_dist[midupperlevels, :, 0] = np.repeat( 
                    0.6 * gaussian(np.sin(lat_center), np.sin(np.deg2rad(20)), self.sin_lats), 
                    len(midupperlevels)).reshape( (self.N_pts, len(midupperlevels)) ).T
                return RH_dist
            self.generate_RH_dist = generate_RH_dist 
            lat_efe = 0   # the latitude (radians) of the EFE. set this as the center of the gaussian
            self.RH_dist = self.generate_RH_dist(lat_efe)

            # # Save RH dists for plotting:
            # np.savez("RH_M0_mebm.npz", RH=self.RH_dist, lats=self.lats, pressures=pressures)
            # np.savez("RH_M18_mebm.npz", RH=self.generate_RH_dist(np.deg2rad(-10.74)), lats=self.lats, pressures=pressures)
            # os.sys.exit()

            # # Debug: Plot RH dist
            # f, ax = plt.subplots(1, figsize=(10,6))
            # levels = np.arange(0, 1.05, 0.05)
            # cf = ax.contourf(self.sin_lats, pressures/100, self.RH_dist[:, :, 0], cmap="BrBG", levels=levels)
            # cb = plt.colorbar(cf, ax=ax, pad=0.1, fraction=0.2)
            # cb.set_ticks(np.arange(0, 1.05, 0.1))
            # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
            # ax.set_yticks(np.arange(0,1001,100))
            # plt.gca().invert_yaxis()
            # plt.show()

            # # Debug: Plot RH 
            # f, ax = plt.subplots(1, figsize=(16,8))
            # # I = 0
            # # ax.plot(self.sin_lats, self.RH_dist[I, :, 0], "b", label="{:3.0f} hPa".format(pressures[I]/100))
            # I = 14
            # ax.plot(self.sin_lats, self.RH_dist[I, :, 0], "r", label="{:3.0f} hPa".format(pressures[I]/100))
            # # I = 23
            # # ax.plot(self.sin_lats, self.RH_dist[I, :, 0], "y", label="{:3.0f} hPa".format(pressures[I]/100))
            # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
            # ax.set_ylim([0, 1])
            # ax.legend(loc="upper left")
            # ax.grid()
            # plt.show()

            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load(EBM_PATH + "/data/moist_adiabat_data.npz")    # load data from a previous moist adiabat calculation using MetPy
            # pressures  = moist_data["pressures"]
            T_surf_sample = moist_data["T_surf_sample"]    # the surface temp points 
            T_data = moist_data["T_data"]    # the resulting vertical levels temps

            pressures_flipped = np.flip(pressures, axis=0)   # sp.interpolate.RectBivariateSpline needs increasing x values
            T_data = np.flip(T_data, axis=1)
            self.interpolated_moist_adiabat = sp.interpolate.RectBivariateSpline(T_surf_sample, pressures_flipped, T_data)    # this returns an object that has the method .ev() to evaluate the interpolation function

            if wv_feedback == False:
                # prescribe WV from control simulation
                T_control = self.ctrl_data["ctrl_state_temp"][0, :, 0]    # control surface temp
                Tgrid_control = np.repeat(T_control, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                air_temp = self.interpolated_moist_adiabat.ev(Tgrid_control, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(air_temp, self.state["air_pressure"].values[:] / 100)[1]    # q = RH * q_sat

            self.pressures = pressures
            self.pressures_flipped = pressures_flipped

            if homog_olr:
                self.ctrl_state_temp = self.ctrl_data["ctrl_state_temp"]
                self.ctrl_state_RH_dist = self.generate_RH_dist(0)
                self.ctrl_state_qstar = self._humidsat(self.ctrl_state_temp, self.state["air_pressure"].values[:] / 100)[1]
                self.ctrl_state_q = self.ctrl_state_RH_dist * self.ctrl_state_qstar

                self.state["air_temperature"].values[:] = self.ctrl_state_temp 
                self.state["surface_temperature"].values[:] = self.ctrl_state_temp[0, :, :]
                self.state["specific_humidity"].values[:] = self.ctrl_state_q
                tendencies, diagnostics = self.longwave_radiation(self.state)
                self.L_ctrl = diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]

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
                if lr_feedback == False:
                    # Retain LR from control simulations by just shifting all levels by difference at surface
                    Tgrid_diff = np.repeat(T - self.ctrl_data["ctrl_state_temp"][0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
                    self.state["air_temperature"].values[:] = self.ctrl_data["ctrl_state_temp"][:] + Tgrid_diff
                else:
                    # Create a 2D array of the T vals and pass to self.interpolated_moist_adiabat
                    Tgrid = np.repeat(T, self.N_levels).reshape( (self.N_pts, self.N_levels) )
                    self.state["air_temperature"].values[:] = self.interpolated_moist_adiabat.ev(Tgrid, pressures).T.reshape( (self.N_levels, self.N_pts, 1) )
                if wv_feedback == True:
                    # Shift RH_dist based on ITCZ
                    E = self.E_dataset[np.searchsorted(self.T_dataset, T)]
                    lat_efe = self.lats[np.argmax(E)] 
                    self.RH_dist = self.generate_RH_dist(lat_efe)
                    # Recalculate q
                    self.state["specific_humidity"].values[:] = self.RH_dist * self._humidsat(self.state["air_temperature"].values[:], self.state["air_pressure"].values[:] / 100)[1]
                tendencies, diagnostics = self.longwave_radiation(self.state)
                if homog_olr == False:
                    return diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
                else:
                    olr = diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
                    dL = olr - self.L_ctrl
                    dL_avg = 1 / self._integrate_lat(1) * self._integrate_lat(dL)
                    return self.L_ctrl + dL_avg
        else:
            os.sys.exit("Invalid keyword for olr_type: {}".format(self.olr_type))

        # save to class
        self.L = L  
        self.homog_olr = homog_olr
        self.lr_feedback = lr_feedback
        self.wv_feedback = wv_feedback
        self.rh_feedback = rh_feedback


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

        if self.al_feedback:
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
            C_mids = self.D_mids / Re**2 * self.dt / self.dx**2 * (1 - self.sin_lats_mids**2)

            row = np.array([])
            col = np.array([])
            data = np.array([])
            for i in range(1, self.N_pts-1):
                row = np.append(row, i)
                col = np.append(col, i - 1)
                data = np.append(data, -C_mids[i-1])

                row = np.append(row, i)
                col = np.append(col, i)
                data = np.append(data, 1 + (C_mids[i] + C_mids[i-1]))
                
                row = np.append(row, i)
                col = np.append(col, i + 1)
                data = np.append(data, -C_mids[i])

            row = np.append(row, [0, 0, self.N_pts-1, self.N_pts-1])
            col = np.append(col, [0, 1, self.N_pts-2, self.N_pts-1])
            data = np.append(data, [1 + C_mids[0], 
                                    -C_mids[0], 
                                    -C_mids[-1], 
                                    1 + C_mids[-1]])

            LHS_matrix = sp.sparse.csc_matrix((data, (row, col)), shape=(self.N_pts, self.N_pts))

            self.step_matrix = sp.sparse.linalg.splu(LHS_matrix)
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
        
        print("\nDiffusivity Type:  {}".format(self.diffusivity))
        print("Insolation Type:   {}".format(self.insolation_type))
        if self.insolation_type == "perturbation":
            print("\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}".format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print("Initial Temp Dist: {}".format(self.initial_condition))
        print("AL Feedback:       {}".format(self.al_feedback))
        print("OLR Scheme:        {}".format(self.olr_type))
        if self.olr_type == "linear":
            print("\tA = {:.2f}, B = {:.2f}".format(self.A, self.B))
        elif "full_radiation" in self.olr_type:
            print("\tHomog OLR:         {}".format(self.homog_olr))
            print("\tLR Feedback:       {}".format(self.lr_feedback))
            print("\tWV Feedback:       {}".format(self.wv_feedback))
            print("\tRH Feedback:       {}".format(self.rh_feedback))
        print("Numerical Method:  {}\n".format(self.numerical_method))
        
        # Setup arrays to save data in
        T_array = np.zeros((frames, self.lats.shape[0]))
        alb_array = np.zeros((frames, self.lats.shape[0]))
        L_array = np.zeros((frames, self.lats.shape[0]))
        
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
            if iteration % its_per_frame == 0:
                error = np.mean(np.abs(self.T - T_array[frame-1, :]) / (its_per_frame * self.dt))
                T_array[frame, :] = self.T
                L_array[frame, :] = self.L(self.T)
                alb_array[frame, :] = self.alb
                # print("Mean alb: {:0.2f}".format(np.mean(self.alb)))

                # Print progress 
                T_avg = np.mean(self.T)
                self._calculate_efe()
                if frame == 0:
                    print("frame = {:5d}; EFE = {:2.3f}; T_avg = {:3.1f}".format(0, np.rad2deg(self.EFE), T_avg))
                else:
                    print("frame = {:5d}; EFE = {:2.3f}; T_avg = {:3.1f}; |dT/dt| = {:.2E}".format(frame, np.rad2deg(self.EFE), T_avg, error))

                # # Debug: Plot RH dist
                # f, ax = plt.subplots(1)
                # levels = np.arange(0, 1.05, 0.05)
                # cf = ax.contourf(self.sin_lats, self.pressures/100, self.RH_dist[:, :, 0], cmap="BrBG", levels=levels)
                # cb = plt.colorbar(cf, ax=ax, pad=0.1, fraction=0.2)
                # cb.set_ticks(np.arange(0, 1.05, 0.1))
                # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
                # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
                # ax.set_yticks(np.arange(0,1001,100))
                # plt.gca().invert_yaxis()
                # plt.show()

                # # Debug: Plot RH 
                # f, ax = plt.subplots(1, figsize=(16,8))
                # I = 14
                # ax.plot(self.sin_lats, self.RH_dist[I, :, 0], "r", label="{:3.0f} hPa".format(self.pressures[I]/100))
                # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
                # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
                # ax.set_ylim([0, 1])
                # ax.legend(loc="upper left")
                # ax.grid()
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

        self.T_f = self.T
        self.E_f = self.E
        self.L_f = self.L(self.T_f)
        self.alb_f = self.alb
        self.S_f = self.S 

        # Truncate arrays
        if iteration-1 % its_per_frame == 0:
            frame += 1
        T_array = T_array[:frame, :]
        alb_array = alb_array[:frame, :]
        L_array = L_array[:frame, :]

        # Print exit messages
        print("Equilibrium reached in {:8.5f} days ({} iterations).".format(iteration * self.dt / self.secs_in_day, iteration))

        if frame == frames:
            print("Failed to reach equilibrium in {:8.5f} days ({} iterations). |dT/dt| = {:4.16f}".format(iteration * self.dt / self.secs_in_day, iteration, error))
        
        print("\nEfficiency: \n{:10.10f} seconds/iteration\n{:10.10f} seconds/sim day\n".format(sim_time / iteration, sim_time / (iteration * self.dt / self.secs_in_day)))

        # Save arrays to class
        self.T_array = T_array
        self.alb_array = alb_array
        self.L_array = L_array



    def save_data(self, control=False):
        """
        Save arrays of state variables.

        INPUTS

        OUTPUTS
        """
        trans_total = self._calculate_trans(self.S_f * (1 - self.alb_f) - self.L(self.T_f), force_zero=True)
        np.savez("simulation_data.npz", T=self.T_array, L=self.L_array, alb=self.alb_array, sin_lats=self.sin_lats, trans_total=trans_total)
        # Save control simulations
        if control:
            L_bar = 1/self._integrate_lat(1) * self._integrate_lat(self.L(self.T_f))
            ctrl_state_temp = self.state["air_temperature"].values[:]
            fname = "ctrl.npz"
            np.savez(fname, S=self.S_f, L_bar=L_bar, trans_total=trans_total, ctrl_state_temp=ctrl_state_temp, alb=self.alb)
            print("{} created".format(fname))


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
        Integrate some array f over x up to index i.

        INPUTS
            f: float, array -> some array or constant to integrate
            i: int -> index in array to integrate up to (default all).

        OUTPUTS
            Returns integral using trapezoidal method.
        """
        if isinstance(f, np.ndarray):
            if i == -1:
                return  2*np.pi*Re**2 * np.trapz(f, dx=self.dx) 
            else:
                return  2*np.pi*Re**2 * np.trapz(f[:i+1], dx=self.dx) 
        else:
            if i == -1:
                return  2*np.pi*Re**2 * np.trapz(f * np.ones(self.N_pts), dx=self.dx) 
            else:
                return  2*np.pi*Re**2 * np.trapz(f * np.ones(self.N_pts)[:i+1], dx=self.dx) 


    def _calculate_trans(self, f, force_zero=False):
        """
        Perform integral calculation to get energy transport.

        INPUTS
            f: array 
            force_zero: boolean

        OUTPUTS
            Returns transport array
        """
        if force_zero:
            area = self._integrate_lat(1)
            f_bar = 1 / area * self._integrate_lat(f)
        if isinstance(f, np.ndarray):
            trans = np.zeros(f.shape)
        else:
            trans = np.zeros(self.N_pts)
        for i in range(self.N_pts):
            if force_zero:
                trans[i] = self._integrate_lat(f - f_bar, i)
            else:
                trans[i] = self._integrate_lat(f, i)
        return trans


    def _calculate_shift(self):
        """
        Calculate approximations of shift in ITCZ using corrent simulation and control values.

        INPUTS

        OUTPUTS
        """
        I_equator = self.N_pts//2 

        # Method 1: Do the basic Taylor approx
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.trans_total, k=4, s=0)
        trans_total_der = spl.derivative()(0)

        numerator = self.trans_total[I_equator]
        denominator = trans_total_der
        shift = -np.rad2deg(np.arcsin(numerator / denominator))
        print("Simple Taylor Shift: {:2.2f}".format(shift))

        # Method 2: Use feedbacks relative to control
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.trans_total_ctrl, k=4, s=0)
        trans_total_ctrl_der = spl.derivative()(0)
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.dtrans_dS, k=4, s=0)
        dtrans_dS_der = spl.derivative()(0)
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.dtrans_pl, k=4, s=0)
        dtrans_pl_der = spl.derivative()(0)
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.dtrans_wv, k=4, s=0)
        dtrans_wv_der = spl.derivative()(0)
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.dtrans_lr, k=4, s=0)
        dtrans_lr_der = spl.derivative()(0)
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, self.dtrans_al, k=4, s=0)
        dtrans_al_der = spl.derivative()(0)

        numerator = self.trans_total_ctrl[I_equator] + self.dtrans_dS[I_equator] + self.dtrans_pl[I_equator] + self.dtrans_wv[I_equator] + self.dtrans_lr[I_equator] + self.dtrans_al[I_equator]
        denominator = trans_total_ctrl_der + dtrans_dS_der + dtrans_pl_der + dtrans_wv_der + dtrans_lr_der + dtrans_al_der 
        shift0 = -np.rad2deg(np.arcsin(numerator / denominator))
        print("Shift with Feedbacks: {:2.2f}".format(shift0))
        
        
        weird_int = self._calculate_trans(1/self._integrate_lat(1)*self._integrate_lat(self.dS*(1-self.alb_f) - self.S_ctrl*(self.alb_f - self.alb_ctrl))- (self.L_bar - self.L_bar_ctrl))
        spl = sp.interpolate.UnivariateSpline(self.sin_lats, weird_int, k=4, s=0)
        weird_int_der = spl.derivative()(0)
        shift = -np.rad2deg(np.arcsin((numerator + weird_int[I_equator]) / (denominator + weird_int_der)))
        percent = 100 * (shift0 - shift) / shift0
        print("With Weird Integral: {:2.2f} ({:3.1f}%)".format(shift, percent))

        shift = -np.rad2deg(np.arcsin((numerator - self.dtrans_pl[I_equator]) / (denominator - dtrans_pl_der)))
        percent = 100 * (shift0 - shift) / shift0
        print("No PL: {:2.2f} ({:3.1f}%)".format(shift, percent))
        shift = -np.rad2deg(np.arcsin((numerator - self.dtrans_wv[I_equator]) / (denominator - dtrans_wv_der)))
        percent = 100 * (shift0 - shift) / shift0
        print("No WV: {:2.2f} ({:3.1f}%)".format(shift, percent))
        shift = -np.rad2deg(np.arcsin((numerator - self.dtrans_lr[I_equator]) / (denominator - dtrans_lr_der)))
        percent = 100 * (shift0 - shift) / shift0
        print("No LR: {:2.2f} ({:3.1f}%)".format(shift, percent))
        shift = -np.rad2deg(np.arcsin((numerator - self.dtrans_al[I_equator]) / (denominator - dtrans_al_der)))
        percent = 100 * (shift0 - shift) / shift0
        print("No AL: {:2.2f} ({:3.1f}%)".format(shift, percent))

        # print("\t- {:1.2E} / {:1.2E}".format(numerator, denominator))


    def log_feedbacks(self, fname_feedbacks):
        """
        Calculate each feedback transport and log data on the shifts.

        INPUTS
            fname_feedbacks: string -> file to write to.

        OUTPUTS
            Creates arrays for each feedback transport saved to class.
        """
        print("\nCalculating feedbacks...")

        self.plot_transports = True

        area = self._integrate_lat(1)
        self.L_bar = 1 / area * self._integrate_lat(self.L_f)

        # Get ctrl data
        ctrl_state_temp = self.ctrl_data["ctrl_state_temp"]
        pert_state_temp = np.copy(self.state["air_temperature"].values[:])

        ctrl_state_RH_dist = self.generate_RH_dist(0)
        ctrl_state_qstar = self._humidsat(ctrl_state_temp, self.state["air_pressure"].values[:] / 100)[1]
        ctrl_state_q = ctrl_state_RH_dist * ctrl_state_qstar
        pert_state_RH_dist = self.generate_RH_dist(self.EFE)
        pert_state_qstar = self._humidsat(pert_state_temp, self.state["air_pressure"].values[:] / 100)[1]
        pert_state_q = np.copy(self.state["specific_humidity"].values[:])

        self.S_ctrl = self.ctrl_data["S"]
        self.alb_ctrl = self.ctrl_data["alb"]
        self.L_bar_ctrl = self.ctrl_data["L_bar"]
        self.dalb = self.alb_f - self.alb_ctrl

        self.trans_total_ctrl = self.ctrl_data["flux_total"]

        self.T_f_ctrl = ctrl_state_temp[0, :, 0]
        self.ctrl_state_temp = ctrl_state_temp
        self.pert_state_temp = pert_state_temp
        self.ctrl_state_q = ctrl_state_q
        self.pert_state_q = pert_state_q

        self.state["air_temperature"].values[:] = ctrl_state_temp 
        self.state["surface_temperature"].values[:] = ctrl_state_temp[0, :, :]
        self.state["specific_humidity"].values[:] = ctrl_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.L_ctrl = diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0]
        self.dL = self.L_f - self.L_ctrl
        
        ## dS
        self.dtrans_dS = self._calculate_trans(self.dS * (1 - self.alb_ctrl), force_zero=True)

        ## dalb
        self.dtrans_dalb = self._calculate_trans((self.S_ctrl + self.dS) * self.dalb, force_zero=True)

        ## Total
        self.trans_total = self._calculate_trans(self.S_f*(1 - self.alb_f) - self.L_f, force_zero=True)
        self.dtrans_total = self.trans_total - self.trans_total_ctrl

        ## Planck
        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
        self.state["air_temperature"].values[:] = pert_state_temp - Tgrid_diff
        self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, :]
        self.state["specific_humidity"].values[:] = pert_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.dL_pl = (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
        self.dtrans_pl = self._calculate_trans(self.dL_pl, force_zero=True)
        
        ## Water Vapor 
        self.state["air_temperature"].values[:] = pert_state_temp
        self.state["surface_temperature"].values[:] = pert_state_temp[0, :, :]
        # self.state["specific_humidity"].values[:] = ctrl_state_q
        self.state["specific_humidity"].values[:] = pert_state_RH_dist * ctrl_state_qstar
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.dL_wv =  (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
        self.dtrans_wv = self._calculate_trans(self.dL_wv, force_zero=True)

        ## Relative Humidity
        self.state["air_temperature"].values[:] = pert_state_temp
        self.state["surface_temperature"].values[:] = pert_state_temp[0, :, :]
        self.state["specific_humidity"].values[:] = ctrl_state_RH_dist * pert_state_qstar
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.dL_rh =  (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
        self.dtrans_rh = self._calculate_trans(self.dL_rh, force_zero=True)
        
        ## Lapse Rate
        Tgrid_diff = np.repeat(pert_state_temp[0, :, 0] - ctrl_state_temp[0, :, 0], self.N_levels).reshape((self.N_pts, self.N_levels)).T.reshape((self.N_levels, self.N_pts, 1))
        self.state["air_temperature"].values[:] =  ctrl_state_temp + Tgrid_diff
        self.state["surface_temperature"].values[:] = self.state["air_temperature"].values[0, :, :]
        self.state["specific_humidity"].values[:] = pert_state_q
        tendencies, diagnostics = self.longwave_radiation(self.state)
        self.dL_lr = (self.L_f - diagnostics["upwelling_longwave_flux_in_air_assuming_clear_sky"].values[-1, :, 0])
        self.dtrans_lr = self._calculate_trans(self.dL_lr, force_zero=True)

        # self._calculate_shift()

        # Calculate lambda
        dS_trans = self._calculate_trans(-self.dS*(1 - self.alb_ctrl), force_zero=True)
        I_equator = self.N_pts//2
        lambda_total = 10**-15 * dS_trans[I_equator] / np.rad2deg(self.EFE)
        print("\nlambda = {:2.2f} PW / deg".format(lambda_total))
        print("1/lambda = {:2.2f} deg / PW".format(1 / lambda_total))


    # def predict_efe(self):
    #     dT = self.T_f - self.T_f_ctrl
    #     lambda_pl = -3.15
    #     lambda_wv = 1.8
    #     lambda_lr = -0.84

    #     # # Feedback contributions
    #     # f, ax = plt.subplots(1, figsize=(16, 10))
    #     # ax.plot(self.sin_lats, dT, "k")
    #     # ax.plot(self.sin_lats, lambda_pl * dT, "r")
    #     # ax.plot(self.sin_lats, lambda_wv * dT, "m")
    #     # ax.plot(self.sin_lats, lambda_lr * dT, "y")
    #     # ax.set_xlabel("Lat")
    #     # ax.set_ylabel("T (K)")
    #     # ax.grid(c="k", ls="--", lw=1, alpha=0.4)
    #     # ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    #     # ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
    #     # plt.tight_layout()
    #     # plt.show()

    #     # Predicted Delta Feedback fluxes
    #     flux_pl = self._calculate_trans(lambda_pl * dT, force_zero=True)
    #     flux_wv = self._calculate_trans(lambda_wv * dT, force_zero=True)
    #     flux_lr = self._calculate_trans(lambda_lr * dT, force_zero=True)
    #     f, ax = plt.subplots(1, figsize=(16, 10))
    #     ax.plot(self.sin_lats, 10**-15 * self.dtrans_total, "k", label="Total")
    #     ax.plot(self.sin_lats, 10**-15 * (self.dtrans_dS + flux_pl + flux_wv + flux_lr), "k--", label="Linear $\\delta S + \\sum \\delta F_i$")
    #     ax.plot(self.sin_lats, 10**-15 * flux_pl, "r--", label="Linear PL")
    #     ax.plot(self.sin_lats, 10**-15 * self.dtrans_pl, "r", label="PL")
    #     ax.plot(self.sin_lats, 10**-15 * flux_wv, "m--", label="Linear WV")
    #     ax.plot(self.sin_lats, 10**-15 * self.dtrans_wv, "m", label="WV")
    #     ax.plot(self.sin_lats, 10**-15 * flux_lr, "y--", label="Linear LR")
    #     ax.plot(self.sin_lats, 10**-15 * self.dtrans_lr, "y", label="LR")
    #     ax.plot(self.sin_lats, 10**-15 * self.dtrans_dS, "c", label="$\\delta S$")
    #     ax.plot(np.sin(self.EFE), 0,  "Xr", label="EFE")
    #     ax.set_xlabel("Latitude")
    #     ax.set_ylabel("Transport [PW]")
    #     ax.grid()
    #     ax.legend()
    #     ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    #     ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
    #     plt.tight_layout()
    #     plt.savefig("linear_feedback_transport.png")
    #     plt.show()
        
    #     # Predicted Flux
    #     f, ax = plt.subplots(1, figsize=(16, 10))
    #     ax.plot(self.sin_lats, 10**-15 * (self.trans_total_ctrl + self.dtrans_dS + flux_pl + flux_wv + flux_lr), "k--", label="Linear $F$")
    #     ax.plot(self.sin_lats, 10**-15 * self.trans_total, "k", label="$F$")
    #     ax.plot(np.sin(self.EFE), 0,  "Xr", label="EFE")
    #     ax.set_xlabel("Lat")
    #     ax.set_ylabel("Transport [PW]")
    #     ax.grid()
    #     ax.legend()
    #     ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
    #     ax.set_xticklabels(["90°S", "", "", "60°S", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "60°N", "", "", "90°N"])
    #     plt.tight_layout()
    #     plt.savefig("linear_transport.png")
    #     plt.show()

    #     # Predicted Shift
    #     dS = self.S_f - self.S_ctrl
    #     dL_bar = self.L_bar - self.L_bar_ctrl

    #     I_equator = self.N_pts//2 
    #     numerator = self.trans_total_ctrl[I_equator] + self._integrate_lat(dS - dL_bar, I_equator) + flux_pl[I_equator] + flux_wv[I_equator] + flux_lr[I_equator]
    #     denominator = 0 

    #     spl = sp.interpolate.UnivariateSpline(self.lats, self.trans_total_ctrl, k=4, s=0)
    #     denominator -= spl.derivative()(0)
    #     spl = sp.interpolate.UnivariateSpline(self.lats, flux_pl, k=4, s=0)
    #     denominator -= spl.derivative()(0)
    #     spl = sp.interpolate.UnivariateSpline(self.lats, flux_wv, k=4, s=0)
    #     denominator -= spl.derivative()(0)
    #     spl = sp.interpolate.UnivariateSpline(self.lats, flux_lr, k=4, s=0)
    #     denominator -= spl.derivative()(0)

    #     shift = np.rad2deg(numerator / denominator)
    #     print("Shift with Feedbacks: {:2.2f}".format(shift))
    #     print("\tNum / Denom = {}/{}".format(numerator, denominator))


    def save_plots(self):
        """
        Plot various data from the simulation

        INPUTS

        OUTPUTS
        """
        ### TEMP 
        print("\nPlotting T")
        
        T_avg = np.mean(self.T_f)
        print("Mean T: {:.2f} K".format(T_avg))

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.T_f, "k", label="Mean $T={:.2f}$ K".format(T_avg))
        ax.set_title("Final Temperature")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("$T$ (K)")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="lower center")
        
        plt.tight_layout()
        
        fname = "temp.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        ### dTEMP 
        print("\nPlotting dT")
        
        T_ctrl = self.ctrl_data["ctrl_state_temp"][0, :, 0]

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.T_f - T_ctrl, "k")
        ax.set_title("Final Temperature Anomaly")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("$T$ (K)")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        
        plt.tight_layout()
        
        fname = "dtemp.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()

        ### MSE
        print("\nPlotting MSE")
        
        self._calculate_efe()
        print("EFE = {:.5f}".format(np.rad2deg(self.EFE)))
        
        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, self.E_f / 1000, "c")
        ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [0, np.max(self.E_f)/1000], "r", label="EFE $\\approx {:.2f}^\\circ$".format(np.rad2deg(self.EFE)))
        ax.set_title("Final Energy")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("MSE (kJ kg$^{-1}$)")
        ax.set_ylim([np.min(self.E_f)/1000 - 1, np.max(self.E_f)/1000 + 1])
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        fname = "mse.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        ### dMSE
        print("\nPlotting dMSE")
        
        E_ctrl = self.E_dataset[np.searchsorted(self.T_dataset, T_ctrl)]
        dE = (self.E_f - E_ctrl) / 1000

        f, ax = plt.subplots(1)
        ax.plot(self.sin_lats, dE, "c")
        ax.plot([np.sin(self.EFE), np.sin(self.EFE)], [np.min(dE) - 100, np.max(dE) + 100], "r", label="EFE $\\approx$ {:.2f}$^\\circ$".format(np.rad2deg(self.EFE)))
        ax.set_title("Final Energy Anomaly")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("MSE (kJ kg$^{-1}$)")
        ax.set_ylim([np.min(dE) - 1, np.max(dE) + 1])
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        fname = "dmse.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()

        ### RADIATION
        print("\nPlotting Radiation")

        S_i = self.S * (1 - self.init_alb)
        L_i = self.L(self.init_temp)
        print("Integral of (S - L): {:.5f} PW".format(10**-15 * self._integrate_lat(self.S_f*(1 - self.alb_f) - self.L_f)))
        
        f, ax = plt.subplots(1)
        l1, = ax.plot(self.sin_lats, self.S_f*(1 - self.alb_f), "r", label="Final $S(1-\\alpha)$")
        l2, = ax.plot(self.sin_lats, S_i, "r--", label="Initial $S(1-\\alpha)$")
        l3, = ax.plot(self.sin_lats, self.L_f, "b", label="Final OLR")
        l4, = ax.plot(self.sin_lats, L_i, "b--", label="Initial OLR")
        l5, = ax.plot(self.sin_lats, self.S_f*(1 - self.alb_f) - self.L_f, "g", label="Final Net")
        l6, = ax.plot(self.sin_lats, S_i - L_i, "g--", label="Initial Net")
        ax.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
        ax.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
        ax.set_ylim([-200, 400])
        ax.set_yticks(np.arange(-200, 401, 50))
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Energy Flux (W m$^{-2})$")
        
        handles = (l1, l2, l3, l4, l5, l6)
        labels = ("Final $S(1-\\alpha)$", "Initial $S(1-\\alpha)$", "Final $L$", "Initial $L$", "Final $NEI$", "Initial $NEI$")
        f.legend(handles, labels, loc="upper center", ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        
        fname = "radiation.png"
        plt.savefig(fname)
        print("{} created.".format(fname))
        plt.close()
        
        if self.plot_transports:
            ### Differences and Transports
            print("\nPlotting Differences and Transports")
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.057, 7.057/1.62/2))

            l1, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb_ctrl), "c")
            l2, = ax1.plot(self.sin_lats, -(self.S_ctrl + self.dS)*self.dalb, "g")
            l3, = ax1.plot(self.sin_lats, -self.dL_pl, "r")
            l4, = ax1.plot(self.sin_lats, -self.dL_wv, "m")
            l5, = ax1.plot(self.sin_lats, -self.dL_rh, "b")
            l6, = ax1.plot(self.sin_lats, -self.dL_lr, "y")
            l7, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb_ctrl) - (self.S_ctrl + self.dS)*self.dalb - self.dL, "k")
            l8, = ax1.plot(self.sin_lats, self.dS*(1 - self.alb_ctrl) - (self.S_ctrl + self.dS)*self.dalb - (self.dL_pl + self.dL_wv + self.dL_rh + self.dL_lr), "k--")
            l9, = ax1.plot(np.sin(self.EFE), 0,  "Xr")

            ax1.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax1.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
            ax1.set_title("(a) Feedback Differences")
            ax1.set_xlabel("Latitude")
            ax1.set_ylabel("Energy Perturbation (W m$^{-2}$)")

            ax2.plot(self.sin_lats, 10**-15 * self.dtrans_dS, "c")
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_dalb, "g")
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_pl, "r")
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_wv, "m")
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_rh, "b")
            ax2.plot(self.sin_lats, -10**-15 * self.dtrans_lr, "y")
            ax2.plot(self.sin_lats, 10**-15 * self.dtrans_total, "k")
            ax2.plot(self.sin_lats, 10**-15 * (self.dtrans_dS - self.dtrans_dalb - (self.dtrans_pl + self.dtrans_wv + self.dtrans_rh + self.dtrans_lr)), "k--")
            ax2.plot(np.sin(self.EFE), 0,  "Xr")

            ax2.set_xticks(np.sin(np.deg2rad(np.arange(-90, 91, 10))))
            ax2.set_xticklabels(["90°S", "", "", "", "", "", "30°S", "", "", "EQ", "", "", "30°N", "", "", "", "", "", "90°N"])
            ax2.set_title("(b) Feedback Transports")
            ax2.set_xlabel("Latitude")
            ax2.set_ylabel("Energy Transport (PW)")

            handles = (l1, l2, l3, l4, l5, l6, l7, l8, l9)
            labels = ("$S'(1 - \\alpha)$", "$-(S + S')\\alpha'$", "$-L_{PL}'$", "$-L_{WV}'$", "$-L_{RH}'$", "$-L_{LR}$",  "$NEI'$", "Sum", "EFE")
            f.legend(handles, labels, loc="upper center", ncol=9)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.80)
            
            fname = "differences_transports.png"
            plt.savefig(fname)
            print("{} created.".format(fname))
            plt.close()

            rc("font", size=7)
            # np.savez("feedback_transports_differences.npz", 
            #     EFE=self.EFE, 
            #     sin_lats=self.sin_lats, 
            #     dtrans_total=self.dtrans_total, 
            #     dtrans_pl=self.dtrans_pl, 
            #     dtrans_wv=self.dtrans_wv, 
            #     dtrans_rh=self.dtrans_rh, 
            #     dtrans_lr=self.dtrans_lr, 
            #     dtrans_dS=self.dtrans_dS, 
            #     dtrans_dalb=self.dtrans_dalb, 
            #     dL_pl=self.dL_pl, 
            #     dL_wv=self.dL_wv, 
            #     dL_rh=self.dL_rh, 
            #     dL_lr=self.dL_lr, 
            #     dL=self.dL,
            #     dS=self.dS,
            #     dalb=self.alb_f - self.alb_ctrl,
            #     alb=self.alb_ctrl,
            #     S=self.S)
