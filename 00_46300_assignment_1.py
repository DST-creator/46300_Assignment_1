#%% Module imports
#General imports
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import math

#Concurrency imports
import ctypes
import concurrent.futures
from multiprocessing import Array as mpArray
from multiprocessing import Value as mpValue

#Optimization code imports
from intersect import intersection
from scipy import optimize

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

#Testing imports
from time import perf_counter

#Custom modules
from _01_utils.utils_46300_assignment_1 import Utils_BEM

#Other imports
import warnings

#%%Global plot settings

#Figure size:
mpl.rcParams['figure.figsize'] = (16, 8)  

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['scatter.marker'] = "+"
mpl.rcParams['lines.color'] = "k"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])
# Cycle through linestyles with color black instead of different colors
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])\
#                                 + mpl.cycler('linestyle', ['-', '--', '-.', ':'])

#Text sizes
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 20

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                 r'\usepackage{siunitx}'])
mpl.rcParams.update({"pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% BEM Calculator
class BEM (Utils_BEM):
    def __init__(self, R = 89.17, P_rtd = 10*1e6, 
                 v_in = 4, v_out = 25, rho=1.225, B=3,
                 airfoil_files=[], bld_file="", t_airfoils = []):
        super().__init__(airfoil_files=airfoil_files, 
                         bld_file=bld_file, 
                         t_airfoils = t_airfoils)
        
        self.R = R              #[m] - Rotor radius 
        self.P_rtd = P_rtd  #[W] - Rated power
        self.rho = rho          #[kg/m^3] - Air density
        self.B = B              #[-] - Number of blades
        self.v_in = v_in        #[m/s] - cut-in velocity
        self.v_out = v_out      #[m/s] - cut-out velocity
        
    def arctan_phi (self, a, a_p, r, tsr=-1, V_0=-1, omega = 1):
        """Calculate the angle between the plane of rotation and the relative 
        velocity via formula 6.7 (cf. script)
        
        Parameters:
            a (scalar numerical value or array-like):
                Axial induction factor a.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            a_p (scalar numerical value or array-like):
                Tangential induction factor a'.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            r (scalar numerical value or array-like):
                Radii at which to calculate the values [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the Turbine [-]
            V_0 (scalar numerical value or array-like):
                Free stream velocity in front of the turbine [m]
                Note: if -1 is used, then the value from the class parameter is
                used
            omega (scalar numerical value or array-like):
                Rotational velocity of the turbine [rad/s]
                Note: if -1 is used, then this parameter is calculated from 
                the tip speed ratio tsr, the free stream velocit V_0 and the 
                rotor radius R
            
        Returns:
            phi (np.float or np.ndarray):
               the angle between the plane of rotation and the relative 
               velocity [rad]
        """
        
        # #Check the input values
        # a, a_p, r, tsr, V_0, omega = self.check_dims (a, a_p, r, tsr, V_0, omega)
        
        if np.any(a>=1) or np.any(a<0):
            raise ValueError("Axial induction factor must be lie within [0,1[")
        
        if np.any(a_p==-1):
            raise ValueError("Tangential induction factor must not be -1")
        
        if not np.any(tsr<=0):
            phi =  np.arctan((np.divide(1-a, 
                                        (1+a_p)*tsr) * self.R/r).astype(float))
        elif not np.any(V_0<=0) and not np.any(omega<=0):
            phi =  np.arctan(np.divide((1-a)*V_0, 
                                       (1+a_p)*omega*r).astype(float))
        else:
            raise ValueError("Invalid input parameters. Either tsr>0 or "
                             + "V_0>0 and omega>0 need to be given")
        
        return phi
    
    def calc_solidity (self, r, c=-1):
        """Calculates the solidity of a blade at a certain radius
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the solidity [m]
        
        Returns:
            sigma (scalar numerical value or array-like):
                Blade solidity at radius r [-]
        """
        
        if c==-1:
            c = np.interp(r, self.bld_df.r, self.bld_df.c)
        
        return (c*self.B)/(2*np.pi*r)

    def calc_ind_factors(self, r, tsr, theta_p = np.pi, 
                         a_0 = 0, a_p_0 = 0, dC_T_0 = 0, 
                         c=-1, tcr=-1, beta=-np.inf, sigma=-1,
                         f=.1, gaulert_method = "classic"):
        """Calulation of the induced velocity factors from the equations of the
        blade element momentum theory.
        The chord length c, thickness to chord ratio tcr, twist angle beta and
        solidity can optionally be provided. This holds the advantage, that 
        they don't need to be recalculated during every iteration of the BEM
        solving.
        
        Parameters:
            r (int or float):
                Radii at which to calculate the values [m]
            tsr (int or float):
                Tip speed ratio of the turbine [-]
            theta_p (int or float):
                Pitch angle of the blades [rad]
                NOTE: Value is assumed to be specified in radians
            a_0 (int or float - optional):
                Start value for the axial induction factor a [-] (default: 0)
            a_p_0  (int or float - optional):
                Start value for the tangential induction factor a' [-] 
                (default: 0)
            dC_T_0 (int or float - optional):
                Start value for infinitesimal thrust coefficient [-] (only 
                needed for Madsen method) (default: 0)
            c (int or float - optional):
                Chord length [m]. If no value is provided, is is calculated 
                from the inputs. 
            tcr (int or float - optional):
                Thickness to chord ratio [-]. If no value is provided, is is calculated 
                from the inputs. 
            beta (int or float - optional):
                Twist angle [rad]. If no value is provided, is is calculated 
                from the inputs. 
            sigma (int or float - optional):
                Solidity [-]. If no value is provided, is is calculated 
                from the inputs. 
            f (float or array-like):
                Relaxation factor (default f=.1)
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
            
        Returns:
            a (float):
                Axial induction factor [-]
            a_p (float):
                Tangential induction factor [-]
            F (float):
                Prandtl correction factor [-]
            dC_T (float):
                Infinitesimal thrust coefficient [-]
        """
        # #Check inputs
        # # Check if the dimensions of the input values match (must be either 
        # # scalar values or array-like of equal shape)
        # r, tsr, theta_p, a_0, a_p_0 = self.check_dims (r, tsr, theta_p, 
        #                                                a_0, a_p_0)

        #Check input for method
        if not type(gaulert_method) == str \
            or gaulert_method not in ['classic', "Madsen"]:
            raise ValueError("Gaulert method must be either 'classic' or "
                             "'Madsen'")
        
        #Check whether some of the values are in the last 2% of the rotor
        # radius. If so, split the values and return a=a_p=-1 for them
        # later on. This region is known to be mathematically unstable
        if r>=.995*self.R:
            return np.zeros(4)
        
        #Interpolate thickness and chord length and beta
        if c == -1:
            c = np.interp(r, self.bld_df.r, self.bld_df.c)
        if tcr == -1:
            tcr = np.interp(r, self.bld_df.r, self.bld_df.tcr)
        if beta == -np.inf:
            beta = np.deg2rad(np.interp(r, self.bld_df.r, self.bld_df.beta))
        
        #Calculate the angle of attack
        phi =  np.arctan((np.divide(1-a_0, 
                                    (1+a_p_0)*tsr) * self.R/r).astype(float))
        theta = theta_p + beta
        
        aoa = phi-theta
        C_l, C_d = self.interp_coeffs (aoa=np.rad2deg(aoa), tcr=tcr)
        
# ######################################
        # C_l = .5
        # C_d = .01
# ######################################
        
        #calculate normal and tangential force factors
        C_n = C_l*np.cos(phi) + C_d*np.sin(phi)
        C_t = C_l*np.sin(phi) - C_d*np.cos(phi)
        
        #Calculate solidity
        if sigma == -1:
            sigma = np.divide(c*self.B, 2*np.pi*r)
        
        #Calculate Prandtl correction factor
        f_corr = self.B/2 * (self.R-r)/(r*np.sin(phi)) 
        F = 2/np.pi * np.arccos(np.exp(-f_corr))
        
        #Calculate induction factors
        #Thrust coefficient (Eq. 6.40)
        dC_T = np.divide(np.power(1-a_0, 2)*C_n*sigma,
                        np.power(np.sin(phi),2))
        
        if gaulert_method == "classic":
            # Temporary axial induction factor
            if a_0 <.33:
                a = np.divide (dC_T,
                               4*F*(1-a_0))
                
                # Full formula in one line (C_T inserted)
                # a_tmp = (sigma*C_n)/(4*F*np.power(np.sin(phi),2)) * (1-a_0)
            else:
                a = np.divide (dC_T,
                               4*F*(1-.25 * (5-3*a_0) * a_0))
            
            # Temporary tangential induction factor
            a_p = (1+a_p_0) * np.divide(sigma*C_t,
                                        4*F*np.sin(phi)*np.cos(phi)) 
            

            
            a = self.relax_parameter(x_tmp=a, x_0=a_0, f=f)
            a_p = self.relax_parameter(x_tmp=a_p, x_0=a_p_0, f=f)
            
        else:
            dC_T = np.divide(np.power(1-a_0, 2)*C_n*sigma,
                            F*np.power(np.sin(phi),2))
            
            dC_T = self.relax_parameter(x_tmp=dC_T, x_0=dC_T_0, f=f)
            
            a = .246*dC_T + .0586*np.power(dC_T,2) + .0883*np.power(dC_T,3)
            
            #Tangential induction factor without corrections (Eq. 6.36):
            a_p = 1 / (np.divide(4*F*np.sin(phi)*np.cos(phi), sigma*C_t) 
                       - 1) 
        
        return a, a_p, F, dC_T
    
    def converge_BEM(self, r, tsr, theta_p = np.pi, 
                     a_0 = 0, a_p_0 = 0, dC_T_0 = 0,
                     c=-1, tcr = -1, beta = -np.inf, sigma = -1,
                     epsilon=1e-6, f = .1, gaulert_method = "classic"):
        """Iterative solver of the equations of blade element momentum theory 
        for the induced velocity factors.
        Note: If the values do not converge within 1000 iterations, the 
        calculation is stopped
        
        Parameters:
            r (int or float):
                Radius at which to calculate the values [m]
            tsr (int or float):
                Tip speed ratio of the turbine 
            theta_p (int or float):
                Pitch angle of the blades [rad]
            a_0 (scalar numerical value or array-like):
                Start value for the axial induction factor a (default: 0)
            a_p_0  (scalar numerical value or array-like):
                Start value for the tangential induction factor a' (default: 0)
            dC_T_0 (scalar numerical value or array-like):
                Start value for dC_T (only needed for Madsen method) (default: 0)
            c (int or float - optional):
                Chord length [m]. If no value is provided, is is calculated 
                from the inputs. 
            tcr (int or float - optional):
                Thickness to chord ratio [-]. If no value is provided, is is calculated 
                from the inputs. 
            beta (int or float - optional):
                Twist angle [rad]. If no value is provided, is is calculated 
                from the inputs. 
            sigma (int or float - optional):
                Solidity [-]. If no value is provided, is is calculated 
                from the inputs. 
            epsilon (float):
                Maximum error tolerance between consecutive iteration 
                (default = 1e-5)
            f (float or array-like):
                Relaxation factor (default f=.1)
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
            
        Returns:
            a (float):
                Axial induction factor
            a_p (float):
                Tangential induction factor
            F (float):
                Prandtl correction factor
            conv_res (float):
                Residual deviation for a and a_p from the last iteration
            n (float):
                Number of iteration
            
        """
        if not np.isscalar(c):
            raise TypeError("Input for c must be scalar")
        elif c == -1:
            c = np.interp(r, self.bld_df.r, self.bld_df.c)
        if not np.isscalar(tcr):
            raise TypeError("Input for tcr must be scalar")
        elif tcr == -1:
            tcr = np.interp(r, self.bld_df.r, self.bld_df.tcr)
        if not np.isscalar(beta):
            raise TypeError("Input for beta must be scalar")
        elif beta == -np.inf:
            beta = np.deg2rad(np.interp(r, self.bld_df.r, self.bld_df.beta))
        if not np.isscalar(sigma):
            raise TypeError("Input for sigma must be scalar")
        elif sigma == -1:
            sigma = np.divide(c*self.B, 2*np.pi*r)
        
########################################################################################
        # tcr = 100    
        # c=1.5
        # beta = np.deg2rad(2)
        # sigma = np.divide(c*self.B, 2*np.pi*r)
########################################################################################
        
        
        
        a, a_p, F, dC_T_0 = self.calc_ind_factors(r=r, tsr=tsr, theta_p=theta_p, 
                                          a_0=a_0, a_p_0=a_p_0, dC_T_0=dC_T_0, 
                                          c=c, tcr = tcr, beta=beta, sigma=sigma,
                                          gaulert_method=gaulert_method)
        n = 1
        
        while (abs(a-a_0)>epsilon) or (abs(a_p-a_p_0)>epsilon): 
            if n>=1000:
                print(f"Maximum iteration number reached before convergence")
                break
            a_0, a_p_0 = a, a_p
            
            #Check validity of a & a_p
            if a_0<0: 
                a_0_corr=0.0
                # print(f"Warning: a<0for r = {r}")
            elif a_0>=1: 
                a_0_corr = .99
                # print(f"Warning: a>1 for r = {r}")
            else:
                a_0_corr = a_0
            
            if a_p==-1:
                a_p_0_corr = -.9999999
                # print(f"Warning: a_p<0 for r = {r}")
            else:
                a_p_0_corr = a_p_0
                
            a, a_p, F, dC_T_0 = self.calc_ind_factors(r=r, tsr = tsr, 
                                                    theta_p=theta_p, 
                                                    a_0=a_0_corr, a_p_0=a_p_0_corr, 
                                                    dC_T_0 = dC_T_0,
                                                    f = f,
                                                    c=c, 
                                                    tcr = tcr, 
                                                    beta=beta, 
                                                    sigma=sigma,
                                                    gaulert_method =
                                                    gaulert_method)

            n +=1
            
        
# =============================================================================
#         if n<=500:
#             print(f"Calculation stopped after {n} iteration")
# =============================================================================
        
# =============================================================================
#         #Print warning for invalid a values
#         if a < 0:
#             print((f"Warning: a=0 for tsr = {tsr}, "+
#                    f"theta_p = {np.rad2deg(theta_p)} deg, r = {r} m"))
#         if a > 1:
#             print((f"Warning: a>=1 for tsr = {tsr}, "+
#                    f"theta_p = {np.rad2deg(theta_p)} deg, r = {r} m"))
#         elif a_p == -1:
#             print((f"Warning: a>=1 for tsr = {tsr}, "+
#                    f"theta_p = {np.rad2deg(theta_p)} deg, r = {r} m"))
# =============================================================================
        
        conv_res = (abs(a-a_0), abs(a_p-a_p_0))
        
        return a, a_p, F, conv_res, n

    def integ_dCp_numerical (self, tsr, theta_p, r_range,
                             c=-1, tcr = -1, beta = -np.inf, sigma = -1,
                             r_range_type = "values", 
                             gaulert_method = "classic"):
        """Integrate the infinitesimal power coefficient over the rotor radius
        
        Parameters:
            r_range (array-like):
                Radii over which to integrate [m]. Can either be specified as by 
                the lower & upper bound and the step width (e.g. [2.8, 89, 1])
                or as discrete radius values
            r_range_type (str):
                Selection whether r_range specifies the bounds and step width 
                or discrete values
            tsr (int or float):
                Tip speed ratio [-]
            theta_p (int or float):
                Pitch angle of the blades [rad]
            c (array-like - optional):
                Chord length for all radii sections [m]. If no value is 
                provided, is is calculated  from the inputs. 
            tcr (array-like - optional):
                Thickness to chord ratio for all radii sections [-]. If no 
                value is provided, is is calculated from the inputs. 
            beta (array-like - optional):
                Twist angle for all radii sections [rad]. If no value is 
                provided, is is calculated from the inputs. 
            sigma (array-like - optional):
                Solidity for all radii sections [-]. If no value is provided, 
                is is calculated from the inputs. 
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
        
        Returns:
            c_p (float):
                Integrated power coefficent [-]
            c_T
                Integrated thrust coefficent [-]
            a_arr (numpy array):
                Axial induction factors for the radii [-]
            a_p_arr (numpy array):
                Tangential induction factors for the radii [-]
            F_arr (numpy array):
                Prandtl correction factors for the radii [-]
        """
        
        #Prepare inputs
        if r_range_type == "bounds":
            r_range, _, _, _ = self.check_radius_range (r_range, self.R)
        elif r_range_type == "values":
            r_range = np.array(r_range)
            if np.any(r_range>self.R) or np.any(r_range<0):
                raise ValueError("All radii in r_range must be within [0,R]")
        else:
            raise ValueError("Invalid value for r_range_type. Must be 'bounds'"
                             " or 'values'")
        
        if type(c)== int and c == -1:
            c = np.array([np.interp(r, self.bld_df.r, self.bld_df.c) 
                          for r in r_range])
        if type(tcr)== int and tcr == -1:
            tcr = np.array([np.interp(r, self.bld_df.r, self.bld_df.tcr) 
                            for r in r_range])
        if type(beta) in [int, float] and beta == -np.inf:
            beta = np.array([np.deg2rad(np.interp(r, 
                                                  self.bld_df.r, 
                                                  self.bld_df.beta)) 
                             for r in r_range])
        if type(sigma)== int and sigma == -1:
            sigma = np.divide(c*self.B, 2*np.pi*np.array(r_range))
        
        #Calculat induction factors
        a_arr = np.array(np.zeros(len(r_range)))
        a_p_arr = np.array(np.zeros(len(r_range)))
        F_arr = np.array(np.zeros(len(r_range)))
        for i,r in enumerate(r_range):
            a_i, a_p_i, F_i, _, _ = self.converge_BEM(r=r, 
                                                    tsr=tsr, 
                                                    c=c[i], tcr=tcr[i], 
                                                    beta=beta[i], 
                                                    sigma=sigma[i],
                                                    theta_p=theta_p,
                                                    gaulert_method =
                                                    gaulert_method)

            a_arr[i] = a_i.item()
            a_p_arr[i] = a_p_i.item()
            F_arr[i] = F_i.item()
        
        #Calculate power coefficient
        dc_p = self.dC_p (r=r_range, tsr=tsr, 
                          a=a_arr, a_p=a_p_arr, theta_p=theta_p)
        dc_T = self.dC_T (r=r_range, a=a_arr, F=F_arr) 
        
        c_p = scipy.integrate.trapezoid(dc_p, r_range)
        c_T = scipy.integrate.trapezoid(dc_T, r_range)
        
        return c_p, c_T, a_arr, a_p_arr, F_arr
    
    def dC_p (self, r, tsr, a=-1, a_p=-1, theta_p=0, 
              gaulert_method = "classic"):
        """Calculatest dC_p (the infinitesimal power coefficient for a annular
        blade ring at the radius r).
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine [-]
            a (scalar numerical value or array-like):
                Axial induction factor a.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            a_p (scalar numerical value or array-like):
                Tangential induction factor a'.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades [rad]
                NOTE: Value is assumed to be specified in radians
            
        Returns:
            dc_p (np.float or np.ndarray): 
                Infinitesimal power coefficient at position r [1/m]
        """
        
# =============================================================================
#         #Check inputs
#         # Check if the dimensions of the input values match (must be either 
#         # scalar values or array-like of equal shape)
#         r, tsr, a, a_p, theta_p = self.check_dims (r, tsr, a, a_p, theta_p)
#         
#         if np.any(r<0) or np.any (r>self.R):
#             raise ValueError("Radius r must be within [0,R]")
#         
#         if np.any(tsr<=0):
#             raise ValueError("Tip speed ratio must be positive and non-zero")
#         
#         if np.any(theta_p<-np.pi) or np.any(theta_p>np.pi):
#             raise ValueError("Pitch must be within [-180, 180] degrees")
# =============================================================================
        
        #Check if a or a_p have a default value. If so, calculate them using BEM
        if any(np.any(var==-1) for var in [a, a_p]):
            a, a_p, F, _, _= self.converge_BEM(r=r, 
                                                tsr=tsr, 
                                                theta_p=theta_p,
                                                gaulert_method = gaulert_method)
        
        
        r = np.array(r)
        #For radii in the last 2% of the rotor radius, the BEM does not return
        #reliable results. This range is therefore neglected and manually set
        #to 0
        i_end = np.where(r>=.995*self.R)
        r_end = r[i_end]
        
        # Radii for which the calculation needs to be performed
        i_valid = np.where(r<.995*self.R)
        r = r[i_valid]
        if np.array(a).size>1: a = a[i_valid]
        if np.array(a_p).size>1: a_p = a_p[i_valid]
        if np.array(tsr).size>1: tsr = tsr[i_valid]
        
        #Calculate dc_p and append 0 for all points in the last 2 % of the 
        #rotor radius
        dc_p = 8*np.power(tsr,2)/(self.R**4)*a_p*(1-a)*np.power(r,3)
        dc_p = np.append (dc_p, np.zeros(r_end.shape))
        
        return dc_p
    
    def dC_T (self, r, a, F):
        """Calculatest dC_T (the infinitesimal thrust coefficient for a annular
        blade ring at the radius r).
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        
        Parameters:
            r (1-D array-like):
                Radii [m]
            a (1-D array-like):
                Axial induction factors [m]
            F (1-D array-like):
                Prandtl correction factors [m]
            
        Returns:
            c_T (float):
                The thrust coefficient
        """
        
        #For radii in the last 2% of the rotor radius, the BEM does not return
        #reliable results. This range is therefore neglected and manually set
        #to 0
        i_end = np.where(r>=.995*self.R)
        r_end = r[i_end]
        
        # Radii for which the calculation needs to be performed
        i_valid = np.where(r<.995*self.R)
        r = r[i_valid]
        if np.array(a).size>1: a = a[i_valid]
        if np.array(F).size>1: F = F[i_valid]
        
        #Calculate dc_p and append 0 for all points in the last 2 % of the 
        #rotor radius
        dc_T = 8/(self.R**2)*r*a*(1-a)*F
        dc_T = np.append (dc_T, np.zeros(r_end.shape))
        
        return dc_T
          
    def integ_dcT (self, r, a, F):
        """Integrates the function for dC_T over r for given values of the 
        radius, the axial induction factor a and the Prandtl correction 
        factor F
        
        Parameters:
            r (1-D array-like):
                Radii [m]
            a (1-D array-like):
                Axial induction factors [m]
            F (1-D array-like):
                Prandtl correction factors [m]
            
        Returns:
            c_T (float):
                The thrust coefficient [-]
        """
        
        c_T = scipy.integrate.trapezoid(8/(self.R**2)*r*a*(1-a)*F, 
                                        r)
        
        return c_T
    
    @staticmethod
    def init_shared_memory(r_range_shared, c_shared, tcr_shared, beta_shared, 
                           sigma_shared, gaulert_method_shared):
        global r_range_global, c_global, tcr_global, beta_global, \
            sigma_global, gaulert_method_global
        
        r_range_global = np.frombuffer(r_range_shared.get_obj(), 
                                       dtype=ctypes.c_double)
        c_global = np.frombuffer(c_shared.get_obj(), 
                                 dtype=ctypes.c_double)
        tcr_global = np.frombuffer(tcr_shared.get_obj(), 
                                   dtype=ctypes.c_double)
        beta_global = np.frombuffer(beta_shared.get_obj(), 
                                    dtype=ctypes.c_double)
        sigma_global = np.frombuffer(sigma_shared.get_obj(), 
                                     dtype=ctypes.c_double)
        gaulert_method_global = gaulert_method_shared.value.decode('utf-8')
     
    def integ_cp_worker (self, tsr, theta_p):
        global r_range_global, c_global, tcr_global, beta_global, \
            sigma_global, gaulert_method_global
        return self.integ_dCp_numerical(tsr=tsr, theta_p=theta_p, 
                                        r_range=r_range_global,  
                                        r_range_type = "values",
                                        c=c_global, tcr=tcr_global, 
                                        beta=beta_global, sigma=sigma_global,
                                        gaulert_method = gaulert_method_global)
    
    def calc_cp (self, tsr_range, theta_p_range, 
                      r_range, r_range_type="values",
                      gaulert_method = "classic",
                      multiprocessing = True):
        """Optimization of the tip speed ratio and pitch angle for the 
        maximization of the power coefficient.
        
        Parameters:
            r_range (array-like):
                Radii over which to integrate [m]. Can either be specified as by 
                the lower & upper bound and the step width (e.g. [2.8, 89, 1])
                or as discrete radius values
            r_range_type (str):
                Selection whether r_range specifies the bounds and step width 
                or discrete values
            tsr_range (array-like):
                Tip speed ratio range to consider
            theta_p_range (array-like):
                Pitch angle range to consider [deg]
                NOTE: Value is assumed to be specified in degrees
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
           multiprocessing (bool - optional):
               Selection whether to use multiprocessing for the BEM solving
               (default: True)
            
            
            
        Returns:
            ds_c (xarray dataset):
                Dataset containing the power and thrust coefficient for all
                combinations of the tip speed and pitch angle range
            ds_bem (xarray dataset):
                Dataset containing the axial and tangential induction 
                coefficient as well as the Prandtl correction factor for all 
                combinations of the tip speed and pitch angle range
        """
        #Prepare inputs
        if r_range_type == "bounds":
            r_range, r_min, r_max, dr = self.check_radius_range (r_range, 
                                                                 self.R)
        elif r_range_type == "values":
            r_range = np.array(r_range)
            if np.any(r_range>self.R) or np.any(r_range<0):
                raise ValueError("All radii in r_range must be within [0,R]")
        else:
            raise ValueError("Invalid value for r_range_type. Must be 'bounds'"
                             " or 'values'")
            
        tsr_range = np.array(tsr_range)
        theta_p_range = np.array(theta_p_range)
        
        #Prepare dataset for the values
        ds_bem = xr.Dataset(
            {},
            coords={"r":r_range, 
                    "tsr":tsr_range,
                    "theta_p":theta_p_range}
            )
        
        ds_bem_shape = [len(r_range), len(tsr_range), len(theta_p_range)]
        
        ds_bem["a"] = (list(ds_bem.coords.keys()),
                       np.empty(ds_bem_shape))
        ds_bem["a_p"] = (list(ds_bem.coords.keys()),
                         np.empty(ds_bem_shape))
        ds_bem["F"] = (list(ds_bem.coords.keys()),
                         np.empty(ds_bem_shape))
        
        ds_c = xr.Dataset(
            {},
            coords={"tsr":tsr_range,
                    "theta_p":theta_p_range}
            )
        
        ds_c_shape = [len(tsr_range), len(theta_p_range)]
        
        ds_c["c_p"] = (list(ds_c.coords.keys()), 
                                np.empty(ds_c_shape))
        ds_c["c_T"] = (list(ds_c.coords.keys()), 
                                np.empty(ds_c_shape))
        
        c = np.array([np.interp(r, self.bld_df.r, self.bld_df.c) 
                      for r in r_range])
        tcr = np.array([np.interp(r, self.bld_df.r, self.bld_df.tcr) 
                        for r in r_range])
        beta = np.deg2rad(np.array([np.interp(r, self.bld_df.r, self.bld_df.beta) 
                         for r in r_range]))
        sigma = np.divide(c*self.B, 2*np.pi*r_range)
        
        if multiprocessing:
            r_range_shared = mpArray(ctypes.c_double, r_range)
            c_shared = mpArray(ctypes.c_double, c)
            tcr_shared = mpArray(ctypes.c_double, tcr)
            beta_shared = mpArray(ctypes.c_double, beta)
            sigma_shared = mpArray(ctypes.c_double, sigma)
            gaulert_method_shared = mpArray(ctypes.c_char, 
                                            gaulert_method.encode('utf-8'))
            
            
            #Multiprocessing with executor.map
            theta_p_comb, tsr_comb = np.meshgrid(theta_p_range, tsr_range)
            theta_p_comb = theta_p_comb.flatten()
            tsr_comb = tsr_comb.flatten()
            
            comb_len = len(tsr_comb)
            csize = int(np.ceil(comb_len/6))
            with concurrent.futures.ProcessPoolExecutor(
                    initializer=self.init_shared_memory, 
                    initargs=(r_range_shared, c_shared, tcr_shared, 
                              beta_shared, sigma_shared, 
                              gaulert_method_shared),
                    max_workers=6) as executor:
                integrator_num = list(executor.map(self.integ_cp_worker,
                                            tsr_comb,
                                            np.deg2rad(theta_p_comb),
                                            chunksize=csize))
                
                for i in range(comb_len):
                    ds_c["c_p"].loc[dict(tsr=tsr_comb[i],
                                           theta_p=theta_p_comb[i])
                                      ] = integrator_num[i][0]
                    ds_c["c_T"].loc[dict(tsr=tsr_comb[i],
                                           theta_p=theta_p_comb[i])
                                      ] = integrator_num[i][1]
                    ds_bem["a"].loc[dict(r = r_range, 
                                         tsr=tsr_comb[i],
                                         theta_p=theta_p_comb[i])
                                    ] = integrator_num[i][2]
                    ds_bem["a_p"].loc[dict(r = r_range, 
                                         tsr=tsr_comb[i],
                                         theta_p=theta_p_comb[i])
                                    ] = integrator_num[i][3]
                    ds_bem["F"].loc[dict(r = r_range, 
                                         tsr=tsr_comb[i],
                                         theta_p=theta_p_comb[i])
                                    ] = integrator_num[i][4]
        else:
            #No Multiprocessing, just Iterating
            for tsr in tsr_range:
                for theta_p in theta_p_range:
                    
                    cp_num, cT_num, a, a_p, F = self.integ_dCp_numerical (
                        tsr=tsr, theta_p=np.deg2rad(theta_p), r_range=r_range,
                        c=c, tcr=tcr, beta=beta, sigma=sigma,
                        r_range_type="values", gaulert_method=gaulert_method)
                    
                    #Save results to dataframe
                    ds_bem["a"].loc[dict(tsr=tsr,theta_p=theta_p)] = a
                    ds_bem["a_p"].loc[dict(tsr=tsr,theta_p=theta_p)] = a_p
                    ds_bem["F"].loc[dict(tsr=tsr,theta_p=theta_p)] = F
                    
                    ds_c["c_p"].loc[dict(tsr=tsr,theta_p=theta_p)] = cp_num
                    ds_c["c_T"].loc[dict(tsr=tsr,theta_p=theta_p)] = cT_num
        
        return ds_c, ds_bem

    def find_c_p_max (self, tsr_lims = [5,10], tsr_step = .5, 
                      theta_p_lims = [-3, 4], theta_p_step = .5,
                      gaulert_method="classic", multiprocessing=True,
                      plot_2d = True, plot_3d = True):
        """Find the approximate tip speed ratio and pitch angle combination 
        within a specified range which results in a local maximum of the power 
        coeffient
        
        Parameters:
            tsr_lims (array-like - optional):
                upper and lower bounds of the considered tip speed ratio range
                (default: [5,10])
            tsr_step (float - optional):
                Resolution / Step width within the tip speed ratio range
                (default: .5)
            theta_p_lims (array-like - optional):
                upper and lower bounds of the considered pitch angle range
                (default: [-3, 4])
            theta_p_step (float - optional):
                Resolution / Step width within the pitch angle range
                (default: .5)
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
           multiprocessing (bool - optional):
               Selection whether to use multiprocessing for the BEM solving
               (default: True)
           plot_2d (bool - optional):
               Selection whether the results should be plotted in a 2d contour 
               plot (default: True)
           plot_3d (bool - optional):
               Selection whether the results should be plotted in a 3d surface
               plot (default: True)
               
        Returns:
            c_p_max (float):
                Local power coefficient maximum within the search boundaries
            tsr_max (float):
                Tip speed ratio at which the local power coefficient maximum 
                occurs
            _ (float):
                Pitch angle at which the local power coefficient maximum 
                occurs
            ds_c (xarray dataset):
                Dataset containing the power and thrust coefficient for all
                combinations of the tip speed and pitch angle range
            ds_bem (xarray dataset):
                Dataset containing the axial and tangential induction 
                coefficient as well as the Prandtl correction factor for all 
                combinations of the tip speed and pitch angle range
        """
        
        r_range = self.bld_df.r
        tsr = np.arange(tsr_lims[0], tsr_lims[1] + tsr_step, tsr_step)
        theta_p=np.arange(theta_p_lims[0], theta_p_lims[1] + theta_p_step , 
                          theta_p_step)
        
        start = perf_counter()
        ds_cp, ds_bem = self.calc_cp (tsr_range=tsr, 
                                      theta_p_range=theta_p,
                                      r_range=r_range,
                                      r_range_type = "values",
                                      gaulert_method=gaulert_method,
                                      multiprocessing=multiprocessing)
        end = perf_counter()
        print (f"C_p_max calculation took {end-start} s")
        
        #Maximum C_P and corresponding coordinates
        c_p_arr= ds_cp["c_p"]
        self.c_p_max = c_p_arr.max().item()
        
        # Find the coordinates where the maximum value occurs
        max_coords = c_p_arr.where(c_p_arr==c_p_arr.max(), drop=True).coords
        # Convert the coordinates to a dictionary for easy access
        coord_dict = {dim: round(coord.values.item(),3) 
                      for dim, coord in max_coords.items()}
        self.tsr_max, self.theta_p_max = coord_dict.values()
        self.c_T_max = float(ds_cp["c_T"].sel(tsr = self.tsr_max, 
                                        theta_p = self.theta_p_max,
                                        method="nearest"))
        
        if plot_2d or plot_3d:
            #Prepare meshgrids
            tsr_mesh, theta_p_mesh = np.meshgrid(tsr, theta_p)
            cp_mesh = ds_cp["c_p"].values.T
            cT_mesh = ds_cp["c_T"].values.T
            tsr_ticks = np.arange(tsr[0], tsr[1] + tsr_step, 1)
            theta_p_ticks = np.arange(theta_p[0], theta_p[1] + theta_p_step, 1)
            label_lst = [r'$\lambda$', r'$\theta_p\:\unit{[\degree]}$']
        
        if plot_2d:
            #Plot C_p
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cp_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                              plt_type="contour", labels=label_lst + [r"$C_p$"],
                              hline=self.theta_p_max, 
                              hline_label=r"$\theta_p=" + str(self.theta_p_max) 
                                          + r"\:\unit{\degree}$",
                              vline=self.tsr_max, 
                              vline_label=r"$\lambda=" 
                                          + str(self.tsr_max) + r"$",
                              intersect_label=r"$C_{p,max}=" + 
                                              str(round(self.c_p_max,3)) + r"$")
            #Plot C_T
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cT_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                              plt_type="contour", labels=label_lst + [r"$C_T$"],
                              hline=self.theta_p_max, 
                              hline_label=r"$\theta_p=" + str(self.theta_p_max) 
                                          + r"\:\unit{\degree}$",
                              vline=self.tsr_max, 
                              vline_label=r"$\lambda=" 
                                          + str(self.tsr_max) + r"$")
        
        if plot_3d:
            #Plot C_p
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cp_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                         plt_type="surface", labels=label_lst + [r"$C_p$"])
            #Plot C_t
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cT_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                         plt_type="surface", labels=label_lst + [r"$C_T$"])
        
        return self.c_p_max, self.tsr_max, self.theta_p_max, ds_cp, ds_bem
    
    def find_v_rtd (self, plot_graphs=True):   
        """Calculation of the rated wind speed associated with the maximum 
        power coefficient. If the maximum power coefficient has not been 
        calculated yet, the calculation for it will automatically started
        
        Parameters:
            plot_graphs(bool - optional):
                Selection whether the results should be plotted (default: True)
            
        Returns:
            None
        """
        
        if not hasattr(self, 'c_p_max'): #I.e. if C_p,max has not been calculated yet
            print("Calculating c_p_max")
            _, _, _, _, _ = self.find_c_p_max(plot_2d=False, plot_3d=False)
        
        #Calculate rated wind speed
        V_0 = np.arange(self.v_in, 12, .25)
        P = self.c_p_max*.5*self.rho*np.pi*self.R**2*V_0**3
        omega = self.tsr_max*V_0/self.R
        self.V_rtd = intersection(V_0, P, 
                               [self.v_in, self.v_out], 
                               [self.P_rtd, self.P_rtd])[0][0]
        self.omega_max = self.tsr_max*self.V_rtd/self.R
        rpm_max = self.omega_max * 60 / (2*np.pi)
        
        if plot_graphs:
            # Plot Power over wind speed
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(V_0, P/1e6, 
                    c="k",
                    label = "Power curve", zorder=3)
            ax.axvline(self.V_rtd, ls="--", lw=1.5, color="k", zorder=2)
            ax.text(self.V_rtd*.97, .2, 
                    "$V_{rated}=" + f"{round(self.V_rtd,2)}"+ r"\:\unit{[\m/\s]}$", 
                    color='k', va='center', ha='center', size = "medium",
                    transform=ax.get_xaxis_transform(), rotation="vertical", 
                    zorder=4)
            ax.axhline(self.P_rtd/1e6, ls="--", lw=1.5, color="k", zorder=2)
            ax.text(0.2, self.P_rtd/1e6*1.03, 
                    "$P_{rated}=" + f"{self.P_rtd/1e6}" + r"\:\unit{MW}$", 
                    color='k', va='center', ha='center', size = "medium",
                transform=ax.get_yaxis_transform(), 
                zorder=4)
            # ax.set_title('Power curve')
            ax.set_xlabel(r'$V_0\:\unit{[\m/\s]}$')
            ax.set_ylabel(r'$P\:\unit{[\MW]}$')
            ax.grid(zorder=1)
            
            fname = "_03_export/Power_curve"
            fig.savefig(fname+".svg")
            fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
            fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
            plt.close(fig)
            
            
            # Plot omega over wind speed
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(V_0, omega, zorder=3)
            ax.axhline(self.omega_max, ls="--", lw=1.5, color="k", zorder=2)
            ax.text(0.2, self.omega_max*1.02, 
                    "$\omega_{max}=" + f"{round(self.omega_max,3)}" + r"\:\unit{rad/\s}$", 
                    color='k', va='bottom', ha='center', size = "medium",
                transform=ax.get_yaxis_transform(), 
                zorder=4)
            ax.axvline(self.V_rtd, ls="--", lw=1.5, color="k", zorder=2)
            ax.text(self.V_rtd*.97, .2, 
                    "$V_{rated}=" + f"{round(self.V_rtd,2)}"+ r"\:\unit{[\m/\s]}$", 
                    color='k', va='center', ha='center', size = "medium",
                    transform=ax.get_xaxis_transform(), rotation="vertical", 
                    zorder=4)
            ax.set_xlabel(r'$V_0\:\unit{[\m/\s]}$')
            ax.set_ylabel(r'$\omega\:\unit{[{rad}/\s]}$')
            ax.grid(zorder=1)
            
            fname = "_03_export/omega_over_V0"
            fig.savefig(fname+".svg")
            fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
            fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
            plt.close(fig)
            
        return self.V_rtd, self.omega_max, rpm_max
    
    def find_pitch_above_rtd (self): 
        """Calculation of the pitch angles in the above rated wind speed region
        which result in a power production of P = 10.64MW.
        The calculation is based on the results for the rated wind speed and 
        maximum power coefficient. If these calculations have not been 
        performed, they will be started automatically.
        In order to accelerate the calculation, an approximation function 
        of the pitch angles based on the wind speed was used. This function 
        consists of a power function with exponent -1 and was fitted to the 
        points V_0 = V_rated, theta_p=0 and V_0 = v_out and the corresponding
        theta_p value (determinded with the BEM). The search region
        for the actual pitch value was chosen to be in a +- 3 deg range
        around the estimated values and is performed in iterative resolution.
        
        Parameters:
            None
        
        Returns:
            df_theta_p (pandas DataFrame):
                Dataframe with the calculated pitch angles for wind speeds in 
                the range [V_rated, v_out] with step width 1 m/s
            theta_p_approx_func (function handle):
                Approximation function of the pitch angle.
        """
        
        if not hasattr(self, 'V_rtd'): #I.e. if V_rated has not been calculated yet
            print("Calculating V_rtd")
            _, _, _,  = self.find_v_rtd(plot_graphs=False)
        
        #Precalculate Blade data
        r_range = self.bld_df.r.to_numpy()
        c = self.bld_df.c.to_numpy()
        tcr = self.bld_df.tcr.to_numpy()
        beta = np.deg2rad(self.bld_df.beta)
        sigma = np.divide(c*self.B, 2*np.pi*r_range)
        
        #pitch angle for cut-out velocity
        tsr = self.omega_max*self.R/self.v_out
        
        #Necessary power coefficent for rated power at v_out
        P=10.64*1e6
        c_p_rtd = P/(.5*self.rho*np.pi*self.R**2*v_out**3)
        
        #Assumed maximum pitch angle: 35 deg (cf. Wind Energy Handbook)
        #Initialize calculation for rough step width
        for theta_p in range(35,0,-2):
            cp, _, a, _, _ = self.integ_dCp_numerical (
                tsr=tsr, theta_p=np.deg2rad(theta_p), 
                r_range=r_range, r_range_type="values",
                c=c, tcr=tcr, beta=beta, sigma=sigma,
                gaulert_method="classic")
            
            if cp>=c_p_rtd:
                break
        
        #Search in smaller radius around the found value
        theta_p_radius_lst = [3, .4, .09]
        dtheta_p_lst = [.5, .1, .01]
        
        for i in range(len(theta_p_radius_lst)):
            theta_p_radius = theta_p_radius_lst[i]
            dtheta_p = dtheta_p_lst[i]
            
            for theta_p in np.arange(theta_p+theta_p_radius,
                                     theta_p-(theta_p_radius+dtheta_p), 
                                     -dtheta_p):
                c_p, _, _, _, _ = self.integ_dCp_numerical (
                    tsr=tsr, theta_p=np.deg2rad(theta_p), 
                    r_range=r_range, r_range_type="values",
                    c=c, tcr=tcr, beta=beta, sigma=sigma,
                    gaulert_method="classic")

                if c_p>=c_p_rtd:
                    break
        
        #Final value for theta_p
        theta_p_out = round(theta_p, 3)
        del theta_p
        
        #Fit a approximation curve to the two points (V_0=V_rated, 
        #theta_p=self.theta_p_max) and (V_0=v_out, theta_p=theta_p_out)
        def power(x, a, b, c=-1.0):
            return a + b * x ** c
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(power, [self.V_rtd, self.v_out],
                                         [self.theta_p_max, theta_p_out], 
                                         p0=[0,0])
        theta_p_approx_func = lambda V_0: power(x=V_0, a=popt[0], b=popt[1])

        #Cacluate residual pitch angles
        #Create wind speed range from V_rtd to v_out-1 m/s
        V_0_range = np.arange(np.ceil(self.V_rtd), self.v_out)
        tsr = self.omega_max*self.R/V_0_range
        n_vels = V_0_range.size
        
        #Necessary power coefficent for rated power at V_0_range
        c_p_rtd = P/(.5*self.rho*np.pi*self.R**2*V_0_range**3)
        
        #Prepare the chord, twist, thickness ratio and solidity
        c = np.array([np.interp(r, self.bld_df.r, self.bld_df.c) 
                      for r in r_range])
        tcr = np.array([np.interp(r, self.bld_df.r, self.bld_df.tcr) 
                        for r in r_range])
        beta = np.deg2rad(np.array([np.interp(r, self.bld_df.r, self.bld_df.beta) 
                         for r in r_range]))
        sigma = np.divide(c*self.B, 2*np.pi*r_range)
        
        r_range_shared = mpArray(ctypes.c_double, r_range)
        c_shared = mpArray(ctypes.c_double, c)
        tcr_shared = mpArray(ctypes.c_double, tcr)
        beta_shared = mpArray(ctypes.c_double, beta)
        sigma_shared = mpArray(ctypes.c_double, sigma)
        gaulert_method_shared = mpArray(ctypes.c_char, 
                                        "classic".encode('utf-8'))
        
        #Prepare theta_p_range for with the approximation function for theta_p
        # Search radius: +-1.5 deg around the estimated value, step .5
        theta_p_est = theta_p_approx_func(V_0_range)
        theta_p_radius = 3
        dtheta_p = .5
        
        theta_p_radius_lst = [3, .4]
        dtheta_p_lst = [.5, .1]
        
        start = perf_counter()
        for i in range(len(theta_p_radius_lst)):
            theta_p_radius = theta_p_radius_lst[i]
            dtheta_p = dtheta_p_lst[i]
            
            theta_p_var = np.arange(-theta_p_radius, theta_p_radius + dtheta_p, 
                                    dtheta_p)                                       #Variation around estimated value
            n_vars = theta_p_var.size
            
            theta_p_mesh = np.tile(theta_p_est.reshape((-1,1)), 
                                       (1,n_vars))
            
            theta_p_mesh = (theta_p_mesh - theta_p_var).flatten()
            tsr_mesh = np.tile(tsr.reshape((-1,1)), 
                               (1,n_vars)).flatten()
            mesh_len = tsr_mesh.size
            
            csize = int(np.ceil(mesh_len/6))
            
            #Parallel processing of variable combinations
            with concurrent.futures.ProcessPoolExecutor(
                    initializer=self.init_shared_memory, 
                    initargs=(r_range_shared, c_shared, tcr_shared, 
                              beta_shared, sigma_shared, 
                              gaulert_method_shared),
                    max_workers=6) as executor:
                
                integrator_num = list(executor.map(self.integ_cp_worker,
                                            tsr_mesh,
                                            np.deg2rad(theta_p_mesh),
                                            chunksize = csize))

            #Retrieve c_p values for each wind velocity and find theta_p which is 
            #closest to c_p_rtd
            for i_v in range(n_vels):
                c_p_i = np.array([integrator_num[i_v*n_vars+j][0] 
                                  for j in range(n_vars)])
                theta_p_i = theta_p_mesh[i_v*n_vars:(i_v+1)*n_vars]
                i = np.asarray(c_p_i<=c_p_rtd[i_v]).nonzero()[0]
                
                if V_0_range[i_v] == 22:
                    pass
                
                if i.size>=1:
                    i_above = i[-1]
                    #Explanation for the index i: First all indices are found for 
                    # which C_p is equal or greater than the C_p,rated. Since the 
                    # rated power increases for pitch angles below the rated pitch 
                    # angle and since the list of pitch angle goes from small to 
                    # large values, the rated pitch angle is the last index that 
                    # was found for this condition
                    
                    if not i_above == theta_p_var.size-1:
                        i_below = i_above+1
                    else:
                        i_below = i_above
                    #Explanation: this is the index of the c_p value which is 
                    #closest and below the C_p,rated
                    
                    #Interpolate between the two pitch angles which were closes to 
                    # the actual pitch angle
                    theta_p_est[i_v] = \
                        round(np.interp(c_p_rtd[i_v],
                                        [c_p_i[i_above], c_p_i[i_below]],
                                        [theta_p_i[i_above], theta_p_i[i_below]]
                                        )
                              , 3)
                    
                else:
                    print("Warning: Pitch angle not found in search range for "
                          + f"V_0={V_0_range[i_v]}")
        
        #Save and return final values
        V_0_ext = np.append(np.insert(V_0_range, 0, self.V_rtd), v_out)
        theta_p_ext = np.append(np.insert(theta_p_est, 0, self.theta_p_max), 
                                theta_p_out)
        self.df_theta_p  = pd.DataFrame(dict(V_0 = V_0_ext,
                                             theta_p = theta_p_ext))
        end = perf_counter()
        print (f"theta_p calculation took {end-start} s")
          
        return self.df_theta_p, theta_p_approx_func

    def aero_power (self, V_in, v_out=-1, V_rtd=-1, c_p_max=-1, theta_p_ar = {},
                    calc_thrust=False, calc_exact=False):
        """Calculates the aerodynamic power for given wind speed values.
        
        
        Parameters:
            V_in (array-like):
                The wind speeds to calculate the power for (in m/s)
            v_out (int or float):
                The cut-out wind speed in m/s
            V_rtd (int or float - optional):
                The rated wind speed in m/s. If this parameter is not specified, 
                the value from the class attribute is used or it is calculated
                using find_v_rtd
            c_p_max (float - optional):
                The power coefficient. If this parameter is not specified, 
                the value from the class attribute is used or it is calculated
                using find_c_p_max
            theta_p_ar (dict or pandas Dataframe - optional):
                The pitch angles for the wind speeds in degrees in the form of
                of a dict or DataFrame with two keys 'V_0' and 'theta_p'. 
                If this parameter is not specified, the values from the class 
                attribute is used or it is calculated using find_pitch_above_rtd.
                
                Not needed if calc_above_rtd is set to False
            calc_exact (Bool - optional):
                Selection whether the power in the above rated region should be
                calculated using the pitch angles and the BEM, or simply set to
                P_rated.
                For wind speeds, where the pitch angles are not explicitly
                given, they are linearly inter-/extrapolated from the given 
                values
            calc_thrust (Bool - optional):
                Selection whether the thrust calculated.
                Note: for this option, 'calc_exact' must be set to True
            
        Returns
            P (numpy array):
                The calculated aerodynamic power of the rotor [W]
            C_p (numpy array):
                The calculated power coefficient of the rotor
            T (numpy array):
                The calculated Thrust of the rotor [N]
            C_T (numpy array):
                The calculated Thrust coefficient of the rotor
        
        """
        
        #Check inputs
        if not set (["V_0", "theta_p"]) == set (theta_p_ar.keys()):     
            if not hasattr(self, 'df_theta_p'): #I.e. if C_p,max has not been calculated yet
                print("Calculating Pitch angles for above rated wind speed region")
                theta_p_ar, _ = self.find_pitch_above_rtd()
            else:
                theta_p_ar = self.df_theta_p
        elif type(theta_p_ar) == dict:
            theta_p_ar = pd.DataFrame(theta_p_ar)
        theta_p_ar.sort_values(by="V_0", ascending=True, inplace=True)
        
        if V_rtd==-1: 
            if not hasattr(self, 'V_rtd'): #I.e. if C_p,max has not been calculated yet
                print("Calculating V_rtd")
                V_rtd, _, _ = self.find_v_rtd(plot_graphs=False)
            else:
                V_rtd = self.V_rtd
        
        if c_p_max==-1:
            if not hasattr(self, 'c_p_max'): #I.e. if C_p,max has not been calculated yet
                print("Calculating c_p_max")
                c_p_max, _, _, _, _ = self.find_c_p_max(plot_2d=False, plot_3d=False)    
            else:
                c_p_max = self.c_p_max
        
        if v_out == -1:
            v_out = self.v_out
        
        if calc_thrust and not calc_exact:
            raise ValueError("For the thrust calculation, the calculation mode"
                             + "must be set to 'exact'")
        
        #Prepare arrays
        V_in = np.array(V_in)
        P = np.full(V_in.shape, self.P_rtd)
        c_p = np.zeros(V_in.shape)
        
        if calc_thrust:
            T = np.zeros(V_in.shape)
            c_T = np.zeros(V_in.shape)
        
        #Below rated
        I_below_rtd = np.logical_and(V_in<V_rtd, V_in>=self.v_in)
        if calc_exact:
            for i in np.argwhere(I_below_rtd).flatten():
                c_p[i], c_T_i, _, _, _= self.integ_dCp_numerical(
                    tsr=self.tsr_max, theta_p=np.deg2rad(self.theta_p_max),
                    r_range = self.bld_df.r)
                
                P[i] = self.available_power(V=V_in[i], rho = self.rho, 
                                            R=self.R) * c_p[i]
                
                if calc_thrust:
                    c_T[i] = c_T_i
                    T[i] = c_T_i*.5*self.rho*np.pi*(self.R**2)*np.power(V_in[i],2)
        else:
           c_p[I_below_rtd]  = c_p_max
           P[I_below_rtd]  = self.available_power(V=V_in[I_below_rtd], 
                                           rho = self.rho, 
                                           R=self.R) * c_p_max
        del I_below_rtd
        
        #Above rated
        if calc_exact:
            I_above_rtd = np.argwhere(np.logical_and(V_in>=V_rtd, 
                                                     V_in<=v_out)).flatten()
            for i in I_above_rtd:
                tsr = self.omega_max*self.R/V_in[i]
                
                theta_p = np.interp(V_in[i], 
                                    theta_p_ar["V_0"], 
                                    theta_p_ar["theta_p"])
                
                c_p[i], c_T_i, _, _, _= self.integ_dCp_numerical(
                    tsr=tsr,  theta_p=np.deg2rad(theta_p),
                    r_range = self.bld_df.r)
                P[i] = self.available_power(V=V_in[i], rho = self.rho, 
                                            R=self.R) * c_p[i]
                if calc_thrust:
                    c_T[i] = c_T_i
                    T[i] = c_T_i*.5*self.rho*np.pi*(self.R**2)*np.power(V_in[i],2)
        else:
            I_above_rtd = np.logical_and(V_in>=V_rtd,V_in<=v_out)
            c_p[I_above_rtd] = self.P_rtd/self.available_power(
                V=V_in[I_above_rtd], rho = self.rho, R=self.R)
        
        #Below cut-in
        I_below_in = V_in<self.v_in
        P[I_below_in] =  0
        del I_below_in

        #Above cut out
        I_above_out = V_in>v_out
        P[I_above_out] =  0
        
        if calc_thrust:
            return P, T, c_p, c_T
        else:
            return P, c_p

    def calc_AEP(self, A=9, k=1.9, v_out=-1, V_rtd=-1, c_p_max=-1):
        """Calculates the annual energy production for a given weibull 
        distribution.
        
        Parameters:
            A (int or float - optional):
                Scale parameter of the weibull distribution (default: 9)
            k (int or float - optional):
                Shape parameter of the weibull distribution (default: 1.9)
            v_out (int or float - optional):
                Cut-out wind speed of the turbine.
                If this parameter is not specified, the values from the class 
                attribute is used
            V_rtd (int or float - optional):
                Rated wind speed of the turbine in m/s
                If this parameter is not specified, the values from the class 
                attribute is used or it is calculated using find_v_rtd
            c_p_max (int or float - optional):
                Maximum power coefficient of the turbine
                If this parameter is not specified, the values from the class 
                attribute is used or it is calculated using find_c_p_max
            
            
        Returns:
            AEP (float):
                The annual energy production in Wh
            P (array-like):
                The power of the turbine for the considered wind speed step
            f_weibull (array-like):
                The probability of each wind speed step
        """
        
        if c_p_max==-1: 
            if not hasattr(self, 'c_p_max'): #I.e. if C_p,max has not been calculated yet
                print("Calculating c_p_max")
                c_p_max, _, _, _, _ = self.find_c_p_max(plot_2d=False, plot_3d=False)
            else:
                c_p_max = self.c_p_max
        
        if V_rtd==-1: 
            if not hasattr(self, 'V_rtd'): #I.e. if C_p,max has not been calculated yet
                print("Calculating V_rtd")
                V_rtd, _, _ = self.find_v_rtd(plot_graphs=False)
            else:
                V_rtd = self.V_rtd
        
        if v_out==-1: v_out = self.v_out
        
        V_range = np.arange(self.v_in, v_out+.1, .1)
        
        f_weibull = np.exp(-np.power(V_range/A, k))
        
        P,_, = self.aero_power (V_in=V_range, v_out=v_out, 
                             V_rtd=V_rtd, c_p_max=c_p_max,
                             calc_exact=False)
        
        AEP = np.sum((P[1:]+P[0:-1])/2 * (f_weibull[0:-1]-f_weibull[1:]) *8760)
        
        return AEP, P, f_weibull

      
    def calc_task_six (self, r=80.14, tsr = 8, 
                       c_bounds = [0,3], dc=.5,
                       theta_p_bounds = [-20, 20], dtheta_p=2,
                       plot_graphs = False):
        for i in range (2):
            c_range = np.arange(c_bounds[0], c_bounds[1]+dc, dc)
            theta_p_range = np.arange(theta_p_bounds[0], theta_p_bounds[1]+dtheta_p,
                                 dtheta_p)
            
            #Prepare dataset for the values
            ds_res = xr.Dataset(
                {},
                coords={"c":c_range,
                        "theta_p":theta_p_range}
                )
            
            ds_res_shape = [len(c_range), len(theta_p_range)]
            
            ds_res["a"] = (list(ds_res.coords.keys()),
                           np.empty(ds_res_shape))
            ds_res["a_p"] = (list(ds_res.coords.keys()),
                           np.empty(ds_res_shape))
            ds_res["dC_p"] = (list(ds_res.coords.keys()),
                           np.empty(ds_res_shape))
            
            for c in c_range:
                sigma = self.calc_solidity(r, c)
                for theta_p in theta_p_range:
                    a, a_p, _, _, _ = self.converge_BEM(r=r, tsr=tsr, 
                                                        theta_p = np.deg2rad(theta_p), 
                                                        c=c, sigma = sigma)
                    dC_p  = 8*np.power(tsr,2)/(self.R**4)*a_p*(1-a)*np.power(r,3)
                    
                    #Save results to dataframe
                    ds_res["a"].loc[dict(c=c,theta_p=theta_p)] = a[0]
                    ds_res["a_p"].loc[dict(c=c,theta_p=theta_p)] = a_p[0]
                    ds_res["dC_p"].loc[dict(c=c,theta_p=theta_p)] = dC_p[0]
            
            # Find the coordinates where the maximum value occurs
            max_item = ds_res["dC_p"].where(ds_res["dC_p"]==ds_res["dC_p"].max(), 
                                            drop=True)
            # Convert the coordinates to a dictionary for easy access
            coord_dict = {dim: round(coord.values.item(),3) 
                          for dim, coord in max_item.coords.items()}
            c_max, theta_p_max = coord_dict.values()
            a_max = ds_res["a"].sel(c=c_max, 
                                    theta_p = theta_p_max, 
                                    method = "nearest").values
            a_p_max = ds_res["a_p"].sel(c=c_max, 
                                    theta_p = theta_p_max, 
                                    method = "nearest").values
            phi_max = self.arctan_phi (a=a_max, a_p=a_p_max, r=r, tsr=tsr)
            
            if plot_graphs:
                #Prepare meshgrids
                c_mesh, theta_p_mesh = np.meshgrid(c_range, theta_p_range)
                cp_mesh = ds_res["dC_p"].values.T
                
                label_lst = [r'$c\:\unit{[\m]}$', 
                             r'$\theta_p\:\unit{[\degree]}$',
                             r"$C_p$"]
                    
                self.plot_3d_data(X=c_mesh, Y=theta_p_mesh, Z=cp_mesh, 
                                  plt_type="contour", 
                                  fname = f"C_p_local_{i}",
                                  labels=label_lst,
                                  hline=theta_p_max, 
                                  hline_label=r"$\theta_p=" + str(theta_p_max) 
                                              + r"\:\unit{\degree}$",
                                  vline=c_max, 
                                  vline_label=r"$c=" + str(c_max) 
                                              + r"\:\unit{\m}$",
                                  intersect_label=r"$C_{p,max}=" + 
                                                  str(round(max_item.item(),4)) 
                                                  + r"$")
            
            #Preparation for Second round
            c_bounds = [c_max-dc,c_max+dc]
            if c_bounds[0]<0: c_bounds[0]=0
            theta_p_bounds = [theta_p_max-dtheta_p, theta_p_max+dtheta_p]
            dc = .1
            dtheta_p=.1
            
            del ds_res
            
            yield c_max, theta_p_max, phi_max
        
    def insert_test_values(self):
        """Insert the previously determined results into the class
        
        Parameters:
            None
        
        Returns:
            None:
        """
        
        self.c_p_max = 0.49891433710148764
        self.tsr_max = 7.5
        self.theta_p_max = .4
        self.V_rtd = 11.169317205114034
        self.omega_max = 0.9394401596765196
        
        V_0 = np.concatenate((np.array([self.V_rtd]), np.arange(12,26)))
        theta_p = [0.0, 5.9, 8.5, 10.4, 12.1, 13.6, 14.9, 16.2, 17.4, 18.5, 19.6, 20.7, 21.7, 22.7, 23.5]
        self.df_theta_p = pd.DataFrame(dict(V_0 = V_0,
                                            theta_p = theta_p))

            


#%% Main    
if __name__ == "__main__":
    R = 89.17
    B = 3 
    P_rtd = 10.64*1e6
    v_in = 4
    v_out = 25
    rho = 1.225

    BEM_calculator =  BEM (R = R,
                           B = B,
                           P_rtd = P_rtd,
                           v_in = v_in,
                           v_out = v_out,
                           rho = rho)
    
    Calc_sel = dict(T1=True,
                    T2=False, 
                    T3=False, 
                    T5=False, 
                    T6=False)
    
    
    #Plot operating range
    # BEM_calculator.test_neg_a(r=41, a_0=.3, a_p_0=0)
#%% Test for Exercise values    

    # R=31
    # tsr = 2.61*R/8
    
    # BEM_calculator =  BEM (R = R,
    #                        B = B,
    #                        P_rtd = P_rtd,
    #                        v_in = v_in,
    #                        v_out = v_out,
    #                        rho = rho)
    

    # a, a_p, F, conv_res, n = BEM_calculator.converge_BEM(r=24.5, 
    #                                         tsr=tsr, 
    #                                         theta_p = np.deg2rad(-3), 
    #                                         a_0 = 0, 
    #                                         a_p_0 = 0, 
    #                                         epsilon=1e-6, 
    #                                         f = .1, 
    #                                         gaulert_method = "Madsen")
    
    
    #%% Task 1
    if Calc_sel ["T1"]:
# =============================================================================
#         #Medium resolution, full search area
#         Calculate C_p values
#         c_P_max, tsr_max, theta_p_max, ds_cp, ds_bem = \
#             BEM_calculator.find_c_p_max(plot_2d=True, plot_3d=True,
#                                         gaulert_method="classic",
#                                         multiprocessing=True)
# =============================================================================
        
        #Fine resolution, full search area
        c_P_max, tsr_max, theta_p_max, ds_cp, ds_bem = \
            BEM_calculator.find_c_p_max(tsr_step = .1, 
                                        theta_p_step = .1,
                                        plot_2d=True, plot_3d=True,
                                        gaulert_method="classic",
                                        multiprocessing=True)
        
# =============================================================================
#         #Fine resolution, only around final value
#         c_P_max, tsr_max, theta_p_max, ds_cp, ds_bem = \
#             BEM_calculator.find_c_p_max(tsr_lims = [7.2,7.7], tsr_step = .1, 
#                                         theta_p_lims = [-.1, .9], theta_p_step = .1,
#                                         plot_2d=True, plot_3d=True,
#                                         gaulert_method="classic",
#                                         multiprocessing=True)
# =============================================================================

#%% Task 2
    if Calc_sel ["T2"]:  
        #Calculate V_rated and omega_max
        V_rtd, omega_max, rpm_max = BEM_calculator.find_v_rtd(plot_graphs=True)

#%% Task 3
    if Calc_sel ["T3"]:
        df_theta_p, approx_func = BEM_calculator.find_pitch_above_rtd()
    
        #Plot the results and compare to values from report
        V_0 = np.arange (4,26)
        theta_p_correct = np.array([2.751, 1.966, 0.896, 0, 0, 0, 0, 0, 4.4502, 
                                    7.266, 9.292, 10.958, 12.499, 13.896, 15.2, 
                                    16.432, 17.618, 18.758, 19.860, 20.927, 21.963, 
                                    22.975])
        
        V_0_rated = np.floor(V_rtd)
        i_rtd = np.where(V_0==V_0_rated)[0][0]
        
        fig, ax = plt.subplots()
        ax.scatter(V_0, theta_p_correct, 
                   label="DTU Wind Energy Report-I-0092", 
                   marker = "+", c="k", zorder=2)
        ax.plot(df_theta_p.V_0, approx_func(df_theta_p.V_0), 
                c="k", ls="--", lw=1.5,
                label="Approximation function", zorder=2)
        V_0_ext = np.concatenate((np.arange(0,np.floor(BEM_calculator.V_rtd)+1), 
                                df_theta_p.V_0.values))
        ax.plot(V_0_ext, 
                np.concatenate((np.full(len(V_0_ext)-len(df_theta_p), 
                               df_theta_p.theta_p[0]), 
                               df_theta_p.theta_p.values)), 
                c="k", ls="-", lw=1.5, marker = "d", ms=6,
                label="BEM calculation", zorder=2)
        ax.grid(zorder=1)
        ax.set_xlabel(r"$V_0\:\unit{[\m/\s]}$")
        ax.set_ylabel(r"$\theta_p\:\unit{[\degree]}$")
        ax.set_xticks(np.arange(0,v_out+1))
        ax.set_yticks(np.arange(0,25))
        
        ax.legend(loc="lower right")
        
        fname = "_03_export/theta_p_above_rated"
        fig.savefig(fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
        plt.close(fig)
        
        
        
        #Plot the full power curve
        # BEM_calculator.insert_test_values()
        V_0 = np.arange (0,27).astype(float)
        V_0_ext = np.insert(V_0, 
                            np.argwhere(V_0>=BEM_calculator.V_rtd)[0][0],
                            BEM_calculator.V_rtd)
        #Inserts points just before v_in and just after v_out
        V_0_ext = np.insert(V_0_ext, 
                            np.argwhere(V_0_ext>BEM_calculator.v_out).flatten()[0],
                            BEM_calculator.v_out*1.001)
        V_0_ext = np.insert(V_0_ext, 
                            np.argwhere(V_0_ext<BEM_calculator.v_in).flatten()[-1]+1,
                            BEM_calculator.v_in*0.999)
        P, T, c_p, c_T = BEM_calculator.aero_power(V_in=V_0_ext,
                                                   calc_thrust=True, 
                                                   calc_exact=True)
        
        #Plot P
        fig, ax = plt.subplots()
        ax.axhline(BEM_calculator.P_rtd*1e-6, c="k", ls="--", lw=1.2, zorder=2)
        ax.plot(V_0_ext, P*1e-6, 
                c="k", ls="-", lw=1.5, zorder=3)
        ax.text(0.2, BEM_calculator.P_rtd*1e-6*0.96, 
                "$P_{rated}=" + f"{np.round(BEM_calculator.P_rtd*1e-6,2)}" + r"\:\unit{MW}$", 
                color='k', va='center', ha='center', size = "medium",
            transform=ax.get_yaxis_transform(), zorder=3)
        
        ax.grid(zorder=1)
        ax.set_xlabel(r"$V_0\:\unit{[\m/\s]}$")
        ax.set_ylabel(r"$P\:\unit{[\MW]}$")
        ax.set_xticks(np.arange(0,27+1))
        
        fname = "_03_export/Power_curve_full"
        fig.savefig(fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
        plt.close(fig)
        
        #Plot T
        fig, ax = plt.subplots()
        ax.plot(V_0_ext, T*1e-6, c="k", ls="-", lw=1.5, zorder=2)
        
        ax.grid(zorder=1)
        ax.set_xlabel(r"$V_0\:\unit{[\m/\s]}$")
        ax.set_ylabel(r"$T\:\unit{[\MN]}$")
        ax.set_xticks(np.arange(0,27+1))
        
        fname = "_03_export/Thrust_curve_full"
        fig.savefig(fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
        plt.close(fig)
        
        #Plot c_p & c_T
        fig, ax1 = plt.subplots()
        plt_c_p = ax1.plot(V_0_ext, c_p, 
                           label = "$C_p$",
                           c="k", ls="-", lw=1.5, zorder=2)
        
        ax1.set_xlabel(r"$V_0\:\unit{[\m/\s]}$")
        ax1.set_ylabel(r"$C_p$")
        ax1.set_xticks(np.arange(0,27+1))
        ax1.set_yticks(np.arange(0,1.1,.1))
        ax1.grid(zorder=1)
    
        ax2 = ax1.twinx()
        plt_c_T = ax2.plot(V_0_ext, c_T, 
                           label = "$C_T$",
                           c="k", ls="--", lw=1.5)
        ax2.set_ylabel(r"$C_T$")
        ax2.set_yticks(np.arange(0,1.1,.1))
        
        lns = plt_c_p+plt_c_T
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        
        fname = "_03_export/Coefficients_curve_full"
        fig.savefig(fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
        plt.close(fig)

#%% Task 5
    if Calc_sel ["T5"]:
        AEP, P, f_weibull = BEM_calculator.calc_AEP(A=9, k=1.9, 
                                                    v_out=-1)
        AEP_20, _ , _ = BEM_calculator.calc_AEP(A=9, k=1.9, 
                                                    v_out=20)
        
        BEM_calculator.plot_weibull()
    
#%% Task 6
    if Calc_sel ["T6"]:
        task_six_gen = BEM_calculator.calc_task_six(plot_graphs = True)
        
        c_max, theta_p_max6, phi_max = next(task_six_gen)
        c_max, theta_p_max6, phi_max = next(task_six_gen)
        phi_max = np.rad2deg(phi_max)
