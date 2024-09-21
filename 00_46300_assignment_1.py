#%% Module imports
#General imports
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import math

#Concurrency imports
import concurrent.futures

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
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
# mpl.rcParams['ztick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for math symbols

#%% BEM Calculator
class BEM (Utils_BEM):
    def __init__(self, R = 89.17, P_rated = 10*1e6, 
                 v_in = 4, v_out = 25, rho=1.225, B=3,
                 airfoil_files=[], bld_file="", t_airfoils = []):
        super().__init__(airfoil_files=airfoil_files, 
                         bld_file=bld_file, 
                         t_airfoils = t_airfoils)
        
        self.R = R              #[m] - Rotor radius 
        self.P_rated = P_rated  #[W] - Rated power
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
    
    def calc_solidity (self, r):
        """Calculates the solidity of a blade at a certain radius
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the solidity [m]
        
        Returns:
            sigma (scalar numerical value or array-like):
                Blade solidity at radius r [-]
        """
        return (self.c*self.B)/(2*np.pi*r)

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
        if np.isscalar(r):
            if r>=.995*self.R:
                return np.zeros(4)
        else:
            # Radii for which p_T and p_N should be zero
            i_end = np.where(r>=.995*self.R)
            r_end = r[i_end]
            
            # Radii for which the calculation needs to be performed
            i_valid = np.where(r<.995*self.R)
            r = r[i_valid]
            if a_0.size>1: a_0 = a_0[i_valid]
            if a_p_0.size>1: a_p_0 = a_p_0[i_valid]
            if tsr.size>1: tsr = tsr[i_valid]
            if theta_p.size>1: theta_p = theta_p[i_valid]
        
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
            
            #Append the radii which were >=.98*R (for these, p_T and p_N should be 
            #zero)
            if not np.isscalar(r):
                a = np.append (a, np.zeros(r_end.shape))
                a_p = np.append (a_p, np.zeros(r_end.shape))
                F = np.append (F, np.zeros(r_end.shape))
            
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
            
            #Append the radii which were >=.98*R (for these, p_T and p_N should be 
            #zero)
            if not np.isscalar(r):
                a = np.append (a, np.zeros(r_end.shape))
                a_p = np.append (a_p, np.zeros(r_end.shape))
                F = np.append (F, np.zeros(r_end.shape))
        
        #Check results for invalid values
        if not np.isscalar(r):
            if any([np.any(a<0 * a>=1), np.any(a_p<0)]):
                print(f"Warning: Invalid a or a_p for r = {r}")
            
            a[a<0] = 0
            a[a>=1] = .99999
            a_p[a_p==-1] = -.9999999
        else:
            if a<0: 
                a=np.array([0])
                # print(f"Warning: a<0for r = {r}")
            elif a>=1: 
                a =np.array([.99])
                # print(f"Warning: a>1 for r = {r}")
            
            if a_p==-1:
                a_p = np.array([-.9999999])
                # print(f"Warning: a_p<0 for r = {r}")
        
        return a, a_p, F, dC_T
    
    def converge_BEM(self, r, tsr, theta_p = np.pi, 
                     a_0 = 0, a_p_0 = 0, dC_T_0 = 0,
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
        c = np.interp(r, self.bld_df.r, self.bld_df.c)
        tcr = np.interp(r, self.bld_df.r, self.bld_df.tcr)
        beta = np.deg2rad(np.interp(r, self.bld_df.r, self.bld_df.beta))
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
            a, a_p, F, dC_T_0 = self.calc_ind_factors(r=r, tsr = tsr, 
                                                    theta_p=theta_p, 
                                                    a_0=a_0, a_p_0=a_p_0, 
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
        
        # #Print warning for invalid a values
        # if a == 0:
        #     print((f"Warning: a=0 for tsr = {tsr}, "+
        #            f"theta_p = {np.rad2deg(theta_p)} deg, r = {r} m"))
        # elif a ==.99:
        #     print((f"Warning: a>=1 for tsr = {tsr}, "+
        #            f"theta_p = {np.rad2deg(theta_p)} deg, r = {r} m"))
        
        conv_res = (abs(a-a_0), abs(a_p-a_p_0))
        
        return a, a_p, F, conv_res, n

    def integ_dCp_numerical (self, tsr, theta_p, r_range,
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

        a_arr = np.array(np.zeros(len(r_range)))
        a_p_arr = np.array(np.zeros(len(r_range)))
        F_arr = np.array(np.zeros(len(r_range)))
        for i,r in enumerate(r_range):
            a_i, a_p_i, F_i, _, _ = self.converge_BEM(r=r, 
                                                    tsr=tsr, 
                                                    theta_p=theta_p,
                                                    gaulert_method =
                                                    gaulert_method)

            a_arr[i] = a_i.item()
            a_p_arr[i] = a_p_i.item()
            F_arr[i] = F_i.item()
        
        dc_p = self.dC_p (r=r_range, tsr=tsr, 
                          a=a_arr, a_p=a_p_arr, theta_p=theta_p)
        dc_T = self.dC_T (r=r_range, a=a_arr, F=F_arr) 
        
        c_p = scipy.integrate.trapezoid(dc_p, r_range)
        c_T = scipy.integrate.trapezoid(dc_T, r_range)
        
        return c_p, c_T, a_arr, a_p_arr, F_arr
    
    def integ_dCp_numerical_madsen (self, tsr, theta_p, r_range,
                                    c=-1, tcr=-1, beta=-np.inf, sigma=-1,
                                    r_range_type = "values"):
        """Partial function to integrade dC_p using the madsen method (needed
        for the multiprocessing)"""
        return self.integ_dCp_numerical (tsr=tsr, theta_p=theta_p, 
                                         r_range=r_range, 
                                         r_range_type = r_range_type,
                                         gaulert_method = "Madsen")
    
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
            a, a_p, F, _, _ = self.converge_BEM(r=r, 
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
        if np.array(a_p.size)>1: a_p = a_p[i_valid]
        if np.array(tsr.size)>1: tsr = tsr[i_valid]
        
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
        if a.size>1: a = a[i_valid]
        if F.size>1: F = F[i_valid]
        
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
            #Multiprocessing with executor.map
            with concurrent.futures.ProcessPoolExecutor() as executor:
                theta_p_comb, tsr_comb = np.meshgrid(theta_p_range, tsr_range)
                theta_p_comb = theta_p_comb.flatten()
                tsr_comb = tsr_comb.flatten()
                
                comb_len = len(tsr_comb)
                
                if gaulert_method == "classic":
                    integrator_num = list(executor.map(self.integ_dCp_numerical,
                                                tsr_comb,
                                                np.deg2rad(theta_p_comb),
                                                [r_range] * comb_len,
                                                [c] * comb_len,
                                                [tcr] * comb_len,
                                                [beta] * comb_len,
                                                [sigma] * comb_len,
                                                np.full(comb_len, "values")))
                else:
                    integrator_num = list(executor.map(self.integ_dCp_numerical_madsen,
                                                tsr_comb,
                                                np.deg2rad(theta_p_comb),
                                                [r_range] * comb_len,
                                                [c] * comb_len,
                                                [tcr] * comb_len,
                                                [beta] * comb_len,
                                                [sigma] * comb_len,
                                                np.full(comb_len, "values")))
                
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
            theta_p_max (float):
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
        
        r_range = BEM_calculator.bld_df.r
        tsr = np.arange(tsr_lims[0], tsr_lims[1] + tsr_step, tsr_step)
        theta_p=np.arange(theta_p_lims[0], theta_p_lims[1] + theta_p_step , 
                          theta_p_step)
        
        start = perf_counter()
        ds_cp, ds_bem = BEM_calculator.calc_cp (tsr_range=tsr, 
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
        
        if plot_2d or plot_3d:
            #Prepare meshgrids
            tsr_mesh, theta_p_mesh = np.meshgrid(tsr, theta_p)
            cp_mesh = ds_cp["c_p"].values.T
            cT_mesh = ds_cp["c_T"].values.T
            tsr_ticks = np.arange(tsr[0], tsr[1] + tsr_step, 1)
            theta_p_ticks = np.arange(theta_p[0], theta_p[1] + theta_p_step, 1)
            label_lst = [r'$\lambda$', r'$\theta_p$']
        
        if plot_2d:
            #Plot C_p
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cp_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                         plt_type="contour", labels=label_lst + [r"$C_p$"])
            #Plot C_T
            self.plot_3d_data(X=tsr_mesh, Y=theta_p_mesh, Z=cT_mesh, 
                              xticks=tsr_ticks, yticks=theta_p_ticks,
                         plt_type="contour", labels=label_lst + [r"$C_T$"])
        
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
                               [P_rated, P_rated])[0][0]
        self.omega_max = self.tsr_max*self.V_rtd/self.R
        rpm_max = self.omega_max * 60 / (2*np.pi)
        
        if plot_graphs:
            # Plot Power over wind speed
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(V_0, P/1e6, label = "Power curve")
            ax.axvline(self.V_rtd, ls="--", lw=1.5, color="k")
            # ax.text(0.2, P_rated/1e6*1.03, '$P_{rated}$', color='k', va='center', ha='center',
            #     transform=ax.get_yaxis_transform())
            ax.axhline(P_rated/1e6, ls="--", lw=1.5, color="k")
            # ax.text(self.V_rated*.98, .2, '$V_{rated}$', color='k', va='center', ha='center',
            #     transform=ax.get_xaxis_transform(), rotation="vertical")
            # ax.set_title('Power curve')
            ax.set_xlabel('$V_0\:[m/s]$')
            ax.set_ylabel('$P\:[MW]$')
            ax.grid()
            
            plt.savefig(fname="./_03_export/Power_curve.svg",
                        bbox_inches = "tight")
            plt.close(fig)
            
            
            # Plot omega over wind speed
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(V_0, omega, label = "Power curve")
            ax.axvline(self.V_rtd, ls="--", lw=1.5, color="k")
            # ax.set_title('$\omega$ over the wind speed')
            ax.set_xlabel('$V_0\:[m/s]$')
            ax.set_ylabel('$\omega\:[{rad}/s]$')
            ax.grid()
            
            plt.savefig(fname="./_03_export/omega_over_V0.svg",
                        bbox_inches = "tight")
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
            
# =============================================================================
#             if np.any(np.where(a==0)):
#                 r_a0 = r_range[np.where(a==0)]
#                 print(f"a=0 found for theta_p = {theta_p}, for "
#                       + f"r = {np.around(r_range[np.where(a==0)],1)}")
# =============================================================================
            
            if cp>=c_p_rtd:
                break
        
        #Search in smaller radius around the found value
        theta_p_radius = 3
        dtheta_p = .5
        for i in range(2):
            for theta_p in np.arange(theta_p+theta_p_radius,
                                     theta_p-(theta_p_radius+dtheta_p), 
                                     -dtheta_p):
                cp, _, _, _, _ = self.integ_dCp_numerical (
                    tsr=tsr, theta_p=np.deg2rad(theta_p), 
                    r_range=r_range, r_range_type="values",
                    c=c, tcr=tcr, beta=beta, sigma=sigma,
                    gaulert_method="classic")
                
    # =============================================================================
    #             if np.any(np.where(a==0)):
    #                 r_a0 = r_range[np.where(a==0)]
    #                 print(f"a=0 found for theta_p = {theta_p}, for "
    #                       + f"r = {np.around(r_range[np.where(a==0)],1)}")
    # =============================================================================
                
                if cp>=c_p_rtd:
                    break
            
            #Search radius for second iteration
            theta_p_radius = .4
            dtheta_p = .1
        
        #Final value for theta_p
        theta_p_max = round(theta_p, 1)
        del theta_p
        
        #Fit a approximation curve to the two points (V_0=V_rated, theta_p=0)
        # and (V_0=v_out, theta_p=theta_p_max)
        def power(x, a, b, c=-1.0):
            return a + b * x ** c
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(power, [self.V_rtd, self.v_out],
                                         [0, theta_p_max], 
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
        
        #Prepare theta_p_range for with the approximation function for theta_p
        # Search radius: +-1.5 deg around the estimated value, step .5
        theta_p_est = theta_p_approx_func(V_0_range)
        theta_p_radius = 3
        dtheta_p = .5
        
        start = perf_counter()
        for i in range(2):
            theta_p_var = np.arange(-theta_p_radius, theta_p_radius + dtheta_p, 
                                    dtheta_p)                                       #Variation around estimated value
            n_vars = theta_p_var.size
            
            theta_p_mesh = np.tile(theta_p_est.reshape((-1,1)), 
                                       (1,n_vars))
            
            theta_p_mesh = (theta_p_mesh - theta_p_var).flatten()
            tsr_mesh = np.tile(tsr.reshape((-1,1)), 
                               (1,n_vars)).flatten()
            mesh_len = tsr_mesh.size

            #Parallel processing of variable combinations
            with concurrent.futures.ProcessPoolExecutor() as executor:
                integrator_num = list(executor.map(self.integ_dCp_numerical,
                                            tsr_mesh,
                                            np.deg2rad(theta_p_mesh),
                                            [r_range] * mesh_len,
                                            [c] * mesh_len,
                                            [tcr] * mesh_len,
                                            [beta] * mesh_len,
                                            [sigma] * mesh_len,
                                            np.full(mesh_len, "values")))
            
            #Retrieve c_p values for each wind velocity and find theta_p which is 
            #closest to c_p_rtd
            for i_v in range(n_vels):
                c_p_i = np.array([integrator_num[i_v*n_vars+j][0] 
                                  for j in range(n_vars)])
                theta_p_i = theta_p_mesh[i_v*n_vars:(i_v+1)*n_vars]
                i = np.asarray(c_p_i<=c_p_rtd[i_v]).nonzero()[0]
                
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
                                        [c_p_i[i_below], c_p_i[i_above]],
                                        [theta_p_i[i_below], theta_p_i[i_above]],
                                        )
                              , 1)
                    
                else:
                    print("Warning: Pitch angle not found in search range for "
                          + f"V_0={V_0_range[i_v]}")
            
            #Change search radius for 2nd iteration
            theta_p_radius = .5
            dtheta_p = .1
        
        #Save and return final values
        V_0_ext = np.append(np.insert(V_0_range, 0, self.V_rtd), v_out)
        theta_p_ext = np.append(np.insert(theta_p_est, 0, 0), theta_p_max)
        self.df_theta_p  = pd.DataFrame(dict(V_0 = V_0_ext,
                                             theta_p = theta_p_ext))
        end = perf_counter()
        print (f"theta_p calculation took {end-start} s")
          
        return self.df_theta_p, theta_p_approx_func
            
        
#%% Main    
if __name__ == "__main__":
    R = 89.17
    B = 3 
    P_rated = 10.64*1e6
    v_in = 4
    v_out = 25
    rho = 1.225

    BEM_calculator =  BEM (R = R,
                           B = B,
                           P_rated = P_rated,
                           v_in = v_in,
                           v_out = v_out,
                           rho = rho)
    
    #Plot operating range
    # BEM_calculator.test_neg_a(r=41, a_0=.3, a_p_0=0)
#%% Test for Exercise values    

    # R=31
    # tsr = 2.61*R/8
    
    # BEM_calculator =  BEM (R = R,
    #                        B = B,
    #                        P_rated = P_rated,
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
    
    #Calculate C_p values
    c_P_max, tsr_max, theta_p_max, ds_cp, ds_bem = \
        BEM_calculator.find_c_p_max(plot_2d=True, plot_3d=True)

#%% Task 2
    
    #Calculate V_rated and omega_max
    V_rtd, omega_max, rpm_max = BEM_calculator.find_v_rtd(plot_graphs=True)

#%% Task 3

    df_theta_p, approx_func = BEM_calculator.find_pitch_above_rtd()

    #Plot the results and compare to values from report
    V_0 = np.arange (4,26)
    theta_p_correct = np.array([2.751, 1.966, 0.896, 0, 0, 0, 0, 0, 4.4502, 
                                7.266, 9.292, 10.958, 12.499, 13.896, 15.2, 
                                16.432, 17.618, 18.758, 19.860, 20.927, 21.963, 
                                22.975])
    
    V_0_rated = 11
    i_rtd = np.where(V_0==V_0_rated)[0][0]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(V_0, theta_p_correct, label="correct values", marker = "+")
    ax.plot(df_theta_p.V_0, df_theta_p.theta_p, 
            label="BEM calculation")
    ax.plot(df_theta_p.V_0, approx_func(df_theta_p.V_0), 
            label="Approximation function")
    ax.grid()
    ax.set_yticks(np.arange(0,25))
    ax.set_xticks(np.arange(4,26))
    plt.savefig("_03_export/theta_p.svg", bbox_inches = "tight")
    plt.close(fig)
    
    
    
    
    