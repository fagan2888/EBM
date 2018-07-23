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
from metpy.calc import moist_lapse
from metpy.units import units
from scipy.integrate import quadrature
from scipy.interpolate import RectBivariateSpline
from time import clock
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from matplotlib import animation, rc

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
    
    def __init__(self, dlat, dtmax_multiple, max_iters, tol):
        self.dlat       = dlat
        self.dy         = np.pi*Re*dlat/180
        self.dtmax      = 0.5 * self.dy**2 / D
        self.dt         = dtmax_multiple * self.dtmax
        self.max_iters  = max_iters
        self.tol        = tol
        self.lats       = np.linspace(-90, 90, int(180/dlat))
        self.T_dataset  = np.arange(100, 400, 1e-3)
        self.q_dataset  = self.humidsat(self.T_dataset, ps/100)[1]
        self.E_dataset  = cp*self.T_dataset + RH*self.q_dataset*Lv


    def humidsat(self, t, p):
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


    def initial_temperature(self, initial_condition, triangle_low=None,
            triangle_high=None):
        self.initial_condition = initial_condition
        if initial_condition == 'triangle':
            self.init_temp = triangle_high - (triangle_high - triangle_low) / 90 * np.abs(self.lats)


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


    def outgoing_longwave(self, olr_type, emissivity=None, A=None, B=None):
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
        #     RH_vals    = moist_data['RH_vals']
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
        #         esat_bl = humidsat(T_atmos[:, bl_indx], pressures[bl_indx] / 100)[0]
        #         optical_depth = 0.622 * k * RH * esat_bl / g
        #         emis = 0.3 + 0.7 * (1 - np.exp(-optical_depth))
        #         return emis * sig * T_atmos[:, up_indx]**4 + (1 - emis) * sig * T**4
            
        elif olr_type in ['full_wvf', 'full_no_wvf']:
            ''' FULL BLOWN '''
            if olr_type == 'full_wvf':
                water_vapor_feedback = True
            else:
                water_vapor_feedback = False
                prescribed_vapor = np.load('data/prescribed_vapor.npz')['arr_0']
        
        
            # Use CliMT radiation scheme along with MetPy's moist adiabat calculator
            nLevels = 30
            radiation = climt.RRTMGLongwave(cloud_overlap_method='clear_only')
            self.state = climt.get_default_state([radiation], x={}, 
                            y={'label' : 'latitude', 'values': self.lats, 'units' : 'degress N'},
                            mid_levels={'label' : 'mid_levels', 'values': np.arange(nLevels), 'units' : ''},
                            interface_levels={'label' : 'interface_levels', 'values': np.arange(nLevels + 1), 'units' : ''}
                            )
        
            # Create the 2d interpolation function: gives function T_moist(T_surf, p)
            moist_data = np.load('data/moist_adiabat_data.npz')
            pressures  = moist_data['pressures']
            Tsample    = moist_data['Tsample']
            Tdata      = moist_data['Tdata']
            RH_vals    = moist_data['RH_vals']

            # RectBivariateSpline needs increasing x values
            pressures = np.flip(pressures, axis=0)
            Tdata = np.flip(Tdata, axis=1)
            interpolated_moist_adiabat = RectBivariateSpline(Tsample, pressures, Tdata)

            if water_vapor_feedback == False:
                self.state['specific_humidity'].values[:, :, :] = prescribed_vapor
        
            def L(T):
                ''' 
                OLR function.
                Outputs OLR given T_surf.
                Assumes moist adiabat structure, uses full blown radiation code from CliMT.
                Sets temp profile with interpolation of moist adiabat calculations from MetPy.
                Sets specific hum profile by assuming constant RH and using humidsat function from Boos
                '''
                # Set surface state
                self.state['surface_temperature'].values[:] = T
                # Create a 2D array of the T vals and pass to interpolated_moist_adiabat
                #   note: shape of 'air_temperature' is (lons, lats, press) 
                Tgrid = np.repeat(T, pressures.shape[0]).reshape( (self.lats.shape[0], pressures.shape[0]) )
                self.state['air_temperature'].values[0, :, :] = interpolated_moist_adiabat.ev(Tgrid, self.state['air_pressure'].values[0, :, :])
                # Set specific hum assuming constant RH
                if water_vapor_feedback == True:
                    self.state['specific_humidity'].values[:] = RH_vals * self.humidsat(self.state['air_temperature'].values[:], 
                                                                    self.state['air_pressure'].values[:] / 100)[1]
                # CliMT takes over here, this is where the slowdown occurs
                tendencies, diagnostics = radiation(self.state)
                return diagnostics['upwelling_longwave_flux_in_air_assuming_clear_sky'].sel(interface_levels=nLevels).values[0]

        self.L = L


    def take_step(self):
        if self.numerical_method == 'euler_for':
            self.E = self.E + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T)) + self.dt * D/dy**2 * ( np.roll(self.E, 1) - 2*self.E + np.roll(self.E,-1) )
        elif self.numerical_method == 'euler_back':
            self.E = self.E + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T)) + self.dt * D/dy**2 * ( np.roll(self.E, 1) - 2*self.E + np.roll(self.E,-1) )
            self.Estar = self.E + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T)) + self.dt * D/dy**2 * ( np.roll(self.E, 1) - 2*self.E + np.roll(self.E,-1) )
        elif self.numerical_method == 'crank':
            self.E = np.dot(self.C, self.E) + self.dt * g/ps * ( (1-self.alb)*self.S - self.L(self.T))
    
        self.E[0] = self.E[1]
        self.E[-1] = self.E[-2]
        self.T = self.T_dataset[np.searchsorted(self.E_dataset, self.E)]
        if self.albedo_feedback:
            self.alb = self.alb_water * np.ones(len(self.lats))
            self.alb[np.where(T <= 273.16)] = self.alb_ice    


    def solve(self, numerical_method, nPlot, nPrint):
        self.numerical_method = numerical_method
        self.nPlot = nPlot
        frames = int(self.max_iters / nPlot)
        if numerical_method == 'crank':
            alpha = 0.5
            
            r = D * self.dt / self.dy**2
            
            A = np.zeros((len(self.lats), len(self.lats)))
            B = np.zeros((len(self.lats), len(self.lats)))
            
            rng = np.arange(len(self.lats)-1)
            
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
            self.C = np.dot(Ainv, B)
        print('\nModel Params:')
        print("dtmax:      {:.2f} s / {:.4f} days".format(self.dtmax, self.dtmax/60/60/24))
        print("dt:         {:.2f} s / {:.4f} days = {:.2f} * dtmax".format(self.dt, self.dt/60/60/24, self.dt/self.dtmax))
        print("dlat:       {:.2f} m".format(self.dlat))
        print("tolerance:  {}".format(self.tol * self.dt))
        print("nPlot:      {}".format(nPlot))
        print("frames:     {}\n".format(frames))
        
        print('Insolation Type:   {}'.format(self.insolation_type))
        if self.insolation_type == 'perturbation':
            print('\tlat0 = {:.0f}, M = {:.0f}, sigma = {:.2f}'.format(
                self.perturb_center, self.perturb_intensity, self.perturb_spread))
        print('Initial Temp Dist: {}'.format(self.initial_condition))
        print('Albedo Feedback:   {}'.format(self.albedo_feedback))
        print('OLR Scheme:        {}'.format(self.olr_type))
        if self.olr_type == 'linear':
            print('\tA = {:.2f}, B = {:.2f}'.format(self.A, self.B))
        print('Numerical Method:  {}\n'.format(self.numerical_method))
        
        T_array   = np.zeros((frames, len(self.lats)))
        E_array   = np.zeros((frames, len(self.lats)))
        alb_array = np.zeros((frames, len(self.lats)))
        L_array   = np.zeros((frames, len(self.lats)))
        
        self.T    = self.init_temp
        self.E    = self.E_dataset[np.searchsorted(self.T_dataset, self.T)]
        self.alb  = self.init_alb
        
        t0          = clock()
        iter_count  = 0
        frame_count = 0
        error       = -1
        while iter_count < self.max_iters:
            if iter_count % nPrint == 0: 
                if iter_count == 0:
                    print('{:5d}/{:.0f} iterations.'.format(iter_count, self.max_iters))
                else:
                    print('{:5d}/{:.0f} iterations. Last error: {:.16f}'.format(iter_count, self.max_iters, error))
            if iter_count % nPlot == 0:
                T_array[frame_count, :]   = self.T
                E_array[frame_count, :]   = self.E
                alb_array[frame_count, :] = self.alb
                L_array[frame_count, :]   = self.L(self.T)
                error                     = np.sum(np.abs(T_array[frame_count, :] - T_array[frame_count-1, :]))
                if error < self.tol * self.dt:
                    frame_count += 1
                    T_array      = T_array[:frame_count, :]
                    E_array      = E_array[:frame_count, :]
                    alb_array    = alb_array[:frame_count, :]
                    L_array      = L_array[:frame_count, :]
                    print('{:5d}/{:.0f} iterations. Last error: {:.16f}'.format(iter_count, self.max_iters, error))
                    print('Equilibrium reached in {} iterations ({:.1f} days).'.format(iter_count, iter_count * self.dt/60/60/24))
                    break
                else:
                    frame_count += 1
            self.take_step()
            iter_count += 1
        tf = clock()
        if T_array.shape[0] == frames:
            print('Failed to reach equilibrium. Final error: {:.16f} K'.format(np.max(np.abs(T_array[-1, :] - T_array[-2, :]))))
        print('\nTime: {:.10f} seconds/iteration\n'.format((tf-t0)/iter_count))
        self.T_array   = T_array
        self.E_array   = E_array
        self.alb_array = alb_array
        self.L_array   = L_array

    def save_data(self):
        # Save data
        np.savez('data/T_array.npz',   self.T_array)
        np.savez('data/E_array.npz',   self.E_array)
        np.savez('data/L_array.npz',   self.L_array)
        np.savez('data/alb_array.npz', self.alb_array)
        if self.olr_type == 'full_wvf':
            prescribed_vapor = self.state['specific_humidity'].values[:, :, :]
            np.savez('data/prescribed_vapor.npz', prescribed_vapor)


    def log_efe(self, fname):
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

        with open('data/' + fname, 'a') as f:
            data = '{:2d}, {:2.2f}, {:2d}, {:2.16f}'.format(self.perturb_center, self.perturb_spread, self.perturb_intensity, closest_root)
            f.write(data + '\n')
            print('Logged "{}" in "{}"'.format(data, fname))

    def save_plots(self):
        ### STYLES
        rc('animation', html='html5')
        rc('lines', linewidth=2, color='b', markersize=10)
        rc('axes', titlesize=20, labelsize=16, xmargin=0.05, ymargin=0.05, 
                linewidth=1.5)
        rc('axes.spines', top=False, right=False)
        rc('xtick', labelsize=13)
        rc('xtick.major', size=5, width=1.5)
        rc('ytick', labelsize=13)
        rc('ytick.major', size=5, width=1.5)
        rc('legend', fontsize=14)

        ### INITIAL DISTRIBUTIONS
        print('\nPlotting Initial Dists')
        fig, ax = plt.subplots(1, figsize=(12,5))
        
        # radiaiton dist
        SW = self.S * (1 - self.init_alb)
        LW = self.L(self.init_temp)
        ax.plot([-90, 90], [0, 0], 'k--', lw=2)
        ax.plot(self.lats, SW, 'r', lw=2, label='SW (with albedo)', alpha=0.5)
        ax.plot(self.lats, LW, 'g', lw=2, label='LW (init)', alpha=0.5)
        ax.plot(self.lats, SW - LW, 'b', lw=2, label='SW - LW', alpha=1.0)
        
        ax.set_title('SW/LW Radiation (init)')
        ax.set_xlabel('Lat')
        ax.set_ylabel('W/m$^2$')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        fname = 'init_rad_dists.png'
        plt.savefig(fname, dpi=120)
        print('{} created.'.format(fname))
        plt.close()
        
        ### FINAL TEMP DIST
        print('\nPlotting Final T Dist')

        print('Mean T: {:.2f} K'.format(np.mean(self.T_array[-1,:])))

        f, ax = plt.subplots(1, figsize=(12,5))
        ax.plot(self.lats, self.T_array[-1, :], 'k')
        ax.set_title("Final Temperature Distribution")
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('T (K)')
        ax.grid(c='k', ls='--', lw=1, alpha=0.4)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        
        plt.tight_layout()
        
        fname = 'final_temp_dists.png'
        plt.savefig(fname, dpi=120)
        print('{} created.'.format(fname))
        plt.close()
        
        
        ### FIND ITCZ
        print('\nPlotting EFE')
        E_f = self.E_array[-1, :] / 1000
        
        spl = UnivariateSpline(self.lats, E_f, k=4, s=0)
        roots = spl.derivative().roots()
        
        max_index = np.argmax(E_f)
        efe_lat = self.lats[max_index]
        
        min_error_index = np.argmin( np.abs(roots - efe_lat) )
        closest_root = roots[min_error_index]
        
        print("Approximate EFE lat = {:.1f}".format(efe_lat))
        print("Root found = {:.16f} (of {})".format(closest_root, len(roots)))
        print("ITCZ = {:.5f}".format(closest_root*0.64))
        
        f, ax = plt.subplots(1, figsize=(12,5))
        ax.plot(self.lats, E_f, 'c', label='Final Energy Distribution', lw=4)
        ax.plot(self.lats, spl(self.lats), 'k--', label='Spline Interpolant')
        min_max = [E_f.min(), E_f.max()]
        ax.plot([efe_lat, efe_lat], min_max, 'm')
        ax.plot([closest_root, closest_root], min_max, 'r')
        ax.text(efe_lat+5, np.average(min_max), "EFE $\\approx$ {:.2f}$^\\circ$".format(closest_root), size=16)
        ax.set_title("Final Energy Distribution")
        ax.legend(fontsize=14, loc="upper left")
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('E (kJ / kg)')
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        
        plt.tight_layout()
        
        fname = 'efe.png'
        plt.savefig(fname, dpi=120)
        print('{} created.'.format(fname))
        plt.close()
        
        ### FINAL RADIATION DIST
        print('\nPlotting Final Radiation Dist')

        T_f = self.T_array[-1, :]
        alb_f = self.alb_array[-1, :]
        SW_f = self.S * (1 - alb_f)
        LW_f = self.L(T_f)
        print('(SW - LW) at EFE: {:.2f} W/m^2'.format(SW_f[max_index] - LW_f[max_index]))

        f, ax = plt.subplots(1, figsize=(12, 5))
        ax.plot(self.lats, SW_f, 'r', label='S(1-$\\alpha$)')
        ax.plot(self.lats, LW_f, 'b', label='OLR')
        ax.plot(self.lats, SW_f - LW_f, 'g', label='Net')
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.grid()
        ax.legend(loc='upper left')
        ax.set_title("Final Radiation Distributions")
        ax.set_xlabel("Latitude (degrees)")
        ax.set_ylabel("W/m$^2$")
        
        plt.tight_layout()
        
        fname = 'final_rad_dist.png'
        plt.savefig(fname, dpi=120)
        print('{} created.'.format(fname))
        plt.close()
        
        
        ### LW vs. T
        print('\nPlotting LW vs. T')
        f, ax = plt.subplots(1, figsize=(8, 5))
        ax.set_title("Relationship Between T$_s$ and OLR")
        ax.set_xlabel("T$_s$ (K)")
        ax.set_ylabel("OLR (W/m$^2$)")
        
        def f(t, a, b):
            return a + b*t
        
        x_data = self.T_array.flatten()
        y_data = self.L_array.flatten()
        
        popt, pcov = curve_fit(f, x_data, y_data)
        print('A: {:.2f} W/m2, B: {:.2f} W/m2/K'.format(popt[0], popt[1]))
        
        xvals = np.linspace(np.min(x_data), np.max(x_data), 1000)
        yvals = f(xvals, *popt)
        
        ax.plot(x_data, y_data, 'co', ms=2, label='data points: "{}"'.format(self.olr_type))
        ax.plot(xvals, f(xvals, *popt), 'k--', label='linear fit')
        ax.text(np.min(xvals), np.mean(yvals), s='A = {:.2f},\nB = {:.2f}'.format(popt[0], popt[1]), size=14)
        ax.legend()
        
        plt.tight_layout()
        
        fname = 'OLR_vs_T_fit_{}.png'.format(self.olr_type)
        plt.savefig(fname, dpi=120)
        print('{} created.'.format(fname))
        plt.close()
        
        # Linear OLR: It works!
        # A: -281.6700000000001 W/m2, B: 1.8000000000000003 W/m2/K
        
        # PLANCK OLR:
        # A: -652.8783145676047 W/m2, B: 3.1022145805413275 W/m2/K
        
        # Full OLR, WV Feedback:
        # A: -417.0478973801873 W/m2, B: 2.349441658553002 W/m2/K
        
        # Full OLR, No WV Feedback:
        # A: -383.62200068534526 W/m2, B: 2.2236068235897157 W/m2/K
        # A: -418.26464942476156 W/m2, B: 2.3554184845083475 W/m2/K    (starting at T_f from WV feedback)
        
        # Shell Somerville:
        # A: -999.1626913063144 W/m2, B: 4.033450693435474 W/m2/K      (k = 0.03)
        # A: -1417.4283920090295 W/m2, B: 5.681866740765462 W/m2/K     (k = 0.20)
        
        ### ANIMATION
        show = 'T'
        # show = 'E'
        # show = 'alb'
        # show = 'L'
        print('\nCreating Animation of {}'.format(show))
        
        # set up the figure, the axis, and the plot element we want to animate
        fig, ax = plt.subplots(1, figsize=(9,5))
        
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel('Latitude (degrees)')
        if show == 'T':
            array = self.T_array
            ax.set_ylabel('T (K)')
        elif show == 'E':
            array = self.E_array
            ax.set_ylabel("W/m$^2$")
        elif show == 'alb':
            array = self.alb_array
            ax.set_ylabel("$\\alpha$")
        elif show == 'L':
            array = self.L_array
            ax.set_ylabel("W/m$^2$")
        ax.set_title('EBM t =  0 days')
        plt.tight_layout(pad=3)
        
        line, = ax.plot(self.lats, array[0, :], 'b')
        
        def init():
            line.set_data(self.lats, array[0, :])
            return (line,)
        
        def animate(i):
            if i%100 == 0: 
                print("{}/{} frames".format(i, len(array)))
            ax.set_title('EBM t = {:.0f} days'.format((i+1)*self.nPlot*self.dt/60/60/24))
            graph = array[i, :]
            line.set_data(self.lats, graph)
            m = graph.min()
            M = graph.max()
            ax.set_ylim([m - 0.01*np.abs(m), M + 0.01*np.abs(M)])
            return line,
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(array), interval=int(8000/len(array)), blit=True)
        
        fname = '{}_anim.mp4'.format(show)
        anim.save(fname)
        print('{} created.'.format(fname))
        
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
