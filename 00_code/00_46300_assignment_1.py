#%% Module imports
#General imports
import scipy
import numpy as np
import xarray as xr
import concurrent.futures
from intersect import intersection

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

#Testing imports
from time import perf_counter

#Custom modules
from utils_46300_assignment_1 import Utils_BEM

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
        
        if np.any(a_p<0):
            raise ValueError("Tangential induction factor must be lie within "
                             + "[0,inf[")
        
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
    
    def relax_parameter (self, x_tmp, x_0, f=.1):
        """Overrelax the variable x by combining the temporary, approximated 
        value x_tmp with the value from the previous iteration x_0 using the 
        relaxation factor f
        
        Parameters:
            x_tmp (float or array-like):
                Temporary, approximated value for x
            x_0 (float or array-like):
                Value of x from the previous iteration
            f (float or array-like):
                Relaxation factor (default f=.1)
            
        Returns:
            x (float or array-like):
                Relaxed value for x
        """
        # #Check inputs
        # # Check if the dimensions of the input values match (must be either 
        # # scalar values or array-like of equal shape)
        # x_tmp, x_0, f = self.check_dims (x_tmp, x_0, f)
        
        #Relax value
        x = f*x_tmp + (1-f)*x_0 
        return x
    
    def calc_ind_factors(self, r, tsr, theta_p = np.pi, 
                         a_0 = 0, a_p_0 = 0, dC_T_0 = 0, 
                         c=-1, tcr=-1, beta=-np.inf, sigma=-1,
                         f=.1, gaulert_method = "classic"):
        """Calulation of the induced velocity factors from the equations of the
        blade element momentum theory.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades [rad]
                NOTE: Value is assumed to be specified in radians
            a_0 (scalar numerical value or array-like):
                Start value for the axial induction factor a (default: 0)
            a_p_0  (scalar numerical value or array-like):
                Start value for the tangential induction factor a' (default: 0)
            dC_T_0 (scalar numerical value or array-like):
                Start value for dC_T (only needed for Madsen method) (default: 0)
            gaulert_method (str):
                Selection of the approach to use for the calculation of the 
                induction factors.
                Possible values:
                - 'classic' (default): Classic practical approximation of the 
                  Gaulert Correction for high values of a
                - 'Madsen': Empirical formula by Madsen et. al.
            
        Returns:
            a (scalar numerical value or array-like):
                Axial induction factor
            a_p (scalar numerical value or array-like):
                Tangential induction factor
            F (scalar numerical value or array-like):
                Prandtl correction factor
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
            if r>=.98*self.R:
                return np.zeros(4)
        else:
            # Radii for which p_T and p_N should be zero
            i_end = np.where(r>=.98*self.R)
            r_end = r[i_end]
            
            # Radii for which the calculation needs to be performed
            i_valid = np.where(r<.98*self.R)
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
        phi = self.arctan_phi (a=a_0, a_p=a_p_0, r=r, tsr=tsr)
        theta = theta_p + beta
        
        aoa = phi-theta
        C_l, C_d = self.interp_coeffs (aoa=np.rad2deg(aoa), tcr=tcr)
        
# ######################################
#         c = 1.5
#         C_l = .5
#         C_d = .01
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
            a_p[a_p<0] = 0
        else:
            if a<0: 
                a=np.array([0])
                # print(f"Warning: a<0for r = {r}")
            elif a>=1: 
                a =np.array([.99])
                # print(f"Warning: a>1 for r = {r}")
            
            if a_p<0:
                a_p = np.array([0])
                # print(f"Warning: a_p<0 for r = {r}")
        
        return a, a_p, F, dC_T
    
    def converge_BEM(self, r, tsr, theta_p = np.pi, 
                     a_0 = 0.3, a_p_0 = 0.0, dC_T_0 = 0,
                     epsilon=1e-4, f = .1, gaulert_method = "classic"):
        """Iterative solver of the equations of blade element momentum theory 
        for the induced velocity factors.
        Note: If the values do not converge within 1000 iterations, the 
        calculation is stopped
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades [rad]
            a_0 (scalar numerical value or array-like):
                Start value for the axial induction factor a (default: 0)
            a_p_0  (scalar numerical value or array-like):
                Start value for the tangential induction factor a' (default: 0)
            dC_T_0 (scalar numerical value or array-like):
                Start value for dC_T (only needed for Madsen method) (default: 0)
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
            a (scalar numerical value or array-like):
                Axial induction factor
            a_p (scalar numerical value or array-like):
                Tangential induction factor
            F (scalar numerical value or array-like):
                Prandtl correction factor
            conv_res (scalar numerical value or array-like):
                Residual deviation for a and a_p from the last iteration
            n (scalar numerical value or array-like):
                Number of iteration
            
        """
        c = np.interp(r, self.bld_df.r, self.bld_df.c)
        tcr = np.interp(r, self.bld_df.r, self.bld_df.tcr)
        beta = np.deg2rad(np.interp(r, self.bld_df.r, self.bld_df.beta))
        sigma = np.divide(c*self.B, 2*np.pi*r)
        
        a, a_p, F, dC_T_0 = self.calc_ind_factors(r=r, tsr=tsr, theta_p=theta_p, 
                                          a_0=a_0, a_p_0=a_p_0, dC_T_0=dC_T_0,
                                          c=c, tcr = tcr, beta=beta, sigma=sigma,
                                          gaulert_method=gaulert_method)
        n = 1
        
        while (abs(a-a_0)>epsilon*a) or (abs(a_p-a_p_0)>epsilon*a_p): 
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
        
        # if n<=500:
        #     print(f"Calculation stopped after {n} iteration")
        
        conv_res = (abs(a-a_0), abs(a_p-a_p_0))
        
        return a, a_p, F, conv_res, n
    
    def calc_local_forces (self, r, tsr, V_0, a=-1, a_p=-1, theta_p=np.pi):
        """Calculates the local forces for given turbine parameters.
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the  [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            V_0 (scalar numerical value or array-like):
                Free flow velocity [m/s] 
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
            p_N (np.float or np.ndarray): 
                Local normal force in N/m
            p_T (np.float or np.ndarray): 
                Local tangential force in N/m
        """
        
        #Check inputs
        # Check if the dimensions of the input values match (must be either 
        # scalar values or array-like of equal shape)
        a, a_p, tsr, theta_p = self.check_dims (a, a_p, tsr, theta_p)
        
        r = np.array(r)
        a = np.array(a)
        a_p = np.array(a_p)
        tsr = np.array(tsr)
        theta_p = np.array(theta_p)
        
        #Check whether some of the values are in the last 2% of the rotor
        # radius. If so, split the values and return p_T=0 and p_N = 0 for them
        # later on. This region is known to be mathematically unstable
        if r.size>1:
            # Radii for which p_T and p_N should be zero
            i_end = np.where(r>=.98*self.R)
            r_end = r[i_end]
            
            # Radii for which the calculation needs to be performed
            i_valid = np.where(r<.98*self.R)
            r = r[i_valid]
            if a.size>1: a = a[i_valid]
            if a_p.size>1: a_p = a_p[i_valid]
            if tsr.size>1: tsr = tsr[i_valid]
            if theta_p.size>1: theta_p = theta_p[i_valid]
        else:
            if r>=.98*self.R:
                return np.float32(0), np.float32(0)
        
        
        #Check if a or a_p have a default value. If so, calculate them using BEM
        if any(np.any(var==-1) for var in [a, a_p]):
            a, a_p, F, _, _ = self.converge_BEM(r=r, tsr=tsr, theta_p=theta_p)
        
        #Check if there are any zero values in a, a_p and tsr (not allowed)
        if not all(np.all(var) for var in [a, a_p, tsr]):
            raise ValueError("No zero values allowed in a, a* and tsr")

        #Calculate the angle of attack
        phi = self.arctan_phi (a=a, a_p=a_p, r=r, tsr=tsr)
        theta = theta_p + self.beta
        
        aoa = phi-theta
        
        #Calculate relative velocity
        omega = tsr*V_0/self.R
        V_rel = np.sqrt((np.power(V_0*(1-a),2) 
                        + np.power(
                            np.multiply(
                                np.multiply(omega,r),
                                (1+a_p)),
                            2)).astype(float)
                        )
        
        #Calculate local lift and drag
        l = .5*self.rho*np.power(V_rel, 2)*self.c*self.C_l
        d = .5*self.rho*np.power(V_rel, 2)*self.c*self.C_d
        
        #Calculate local normal and tangential forces
        p_N = l*np.cos(phi) + d*np.sin(phi)
        p_T = l*np.sin(phi) - d*np.cos(phi)
        
        #Append the radii which were >=.98*R (for these, p_T and p_N should be 
        #zero)
        if r.size>1:
            p_N = np.append (p_N, np.zeros(r_end.shape))
            p_T = np.append (p_T, np.zeros(r_end.shape))
        
        return p_N, p_T
    
    def calc_local_tang_force (self, r, tsr, V_0, a=-1, a_p=-1, theta_p=np.pi):
        """Calculates the local tangential force for given turbine parameters.
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values  [m]
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            V_0 (scalar numerical value or array-like):
                Free flow velocity [m/s]
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
            p_T (np.float or np.ndarray): 
                Local tangential force in N/m
        """
        
        p_N, p_T = self.calc_local_forces (r, tsr, a=-1, a_p=-1, theta_p=np.pi)
        
        return p_T
    
    def integ_dCp_numerical (self, tsr, theta_p, r_range, 
                             r_range_type = "values", 
                             gaulert_method = "classic"):
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
    
    def integ_dCp_analytical (self, tsr, theta_p, r_min=.1, r_max=-1, 
                              gaulert_method = "classic"):
        if not all([np.isscalar(var) for var in [tsr, theta_p, r_min, r_max]]):
            raise TypeError("Input values must be scalar")
        
        if r_min< 0 or r_min>self.R:
            raise ValueError("Lower bound must be within [0,R]")
        
        if r_max == -1:
            r_max = self.R
        elif r_min>r_max:
            raise ValueError("Upper bound must be higher than lower bound")
        elif r_max>self.R:
            raise ValueError("Upper bound must be within [0,R]")
        
        if tsr <=0:
            raise ValueError("Tip speed ratio must be positive and non-zero")
    
        c_p, c_p_abserr = scipy.integrate.quad(self.dC_p, 
                                r_min, r_max,
                                args = (tsr, -1, -1, theta_p, gaulert_method))
        
        return c_p, c_p_abserr
    
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
            V_0 (scalar numerical value or array-like):
                Free flow velocity [m/s]
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
        
        #For radii in the last 2% of the rotor radius, the BEM does not return
        #reliable results. This range is therefore neglected and manually set
        #to 0
        i_end = np.where(r>=.98*self.R)
        r_end = r[i_end]
        
        # Radii for which the calculation needs to be performed
        i_valid = np.where(r<.98*self.R)
        r = r[i_valid]
        if a.size>1: a = a[i_valid]
        if a_p.size>1: a_p = a_p[i_valid]
        if tsr.size>1: tsr = tsr[i_valid]
        
        #Calculate dc_p and append 0 for all points in the last 2 % of the 
        #rotor radius
        dc_p = 8*np.power(tsr,2)/(self.R**4)*a_p*(1-a)*np.power(r,3)
        dc_p = np.append (dc_p, np.zeros(r_end.shape))
        
        return dc_p
    
    def dC_T (self, r, a, F):
        """Calculatest dC_p (the infinitesimal power coefficient for a annular
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
        i_end = np.where(r>=.98*self.R)
        r_end = r[i_end]
        
        # Radii for which the calculation needs to be performed
        i_valid = np.where(r<.98*self.R)
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
                The thrust coefficient
        """
        
        c_T = scipy.integrate.trapezoid(8/(self.R**2)*r*a*(1-a)*F, 
                                        r)
        
        return c_T
        
    
    def integ_M_analytical (self, tsr, theta_p, r_min=.1, r_max=-1):
        if not all([np.isscalar(var) for var in [tsr, theta_p, r_min, r_max]]):
            raise TypeError("Input values must be scalar")
        
        if r_min< 0 or r_min>self.R:
            raise ValueError("Lower bound must be within [0,R]")
        
        if r_max == -1:
            r_max = self.R
        elif r_min>r_max:
            raise ValueError("Upper bound must be higher than lower bound")
        elif r_max>self.R:
            raise ValueError("Upper bound must be within [0,R]")
        
        if tsr <=0:
            raise ValueError("Tip speed ratio must be positive and non-zero")
        
        M_integ, M_abserr = scipy.integrate.quad(self.calc_local_tang_force, 
                                r_min, r_max,
                                args = (tsr, -1, -1, theta_p))
        
        return M_integ, M_abserr
    
    def integ_M_numerical (self, tsr, theta_p, 
                           r_min = .1, r_max = -1, dr = .1):
        #Prepare inputs
        if r_min< 0 or r_min>self.R:
            raise ValueError("Lower bound must be within [0,R]")
        
        if r_max == -1:
            r_max = self.R
        elif r_min>r_max:
            raise ValueError("Upper bound must be higher than lower bound")
        elif r_max>self.R:
            raise ValueError("Upper bound must be within [0,R]")
            
        if dr< 0:
            raise ValueError("Step width must be positive")
        elif dr>r_max-r_min:
            raise ValueError("Step width must be smaller than interval of"
                             + "r_min and r_max")
        
        r_range = np.arange(r_min, r_max, dr)
        
        a_arr = a_p_arr = np.array(np.zeros(len(r_range)))
        
        for i,r in enumerate(r_range):
            a_arr[i], a_p_arr[i], _, _, _ = self.converge_BEM(r, theta_p)
        
        p_N, p_T = self.calc_local_forces (
            r=r_range, 
            tsr=tsr,
            a=a_arr,
            a_p=a_p_arr, 
            theta_p=theta_p
            )
        
        M_num = scipy.integrate.trapezoid(p_T, r_range)
        
        return M_num, p_T, a_arr, a_p_arr
    
    def inv_c_p (self, x, r_range, r_range_type="values", 
                      gaulert_method = "classic", 
                      solver = "num"):
        
        #Prepare inputs
        if r_range_type == "bounds":
            r_range, r_min, r_max, dr = self.check_radius_range (r_range, 
                                                                 self.R)
        elif r_range_type == "values":
            if solver == "anal":
                raise TypeError("For analytical solving, r_range_type must be"
                                " 'bounds' and the boundaries of r must be "
                                "specified in r_range")
            
            r_range = np.array(r_range)
            if np.any(r_range>self.R) or np.any(r_range<0):
                raise ValueError("All radii in r_range must be within [0,R]")
        else:
            raise ValueError("Invalid value for r_range_type. Must be 'bounds'"
                             " or 'values'")
        
        tsr, theta_p = x
        
        if solver == "num":
            c_p, _,_,_ = self.integ_dCp_numerical(tsr, theta_p, r_range, 
                                                  r_range_type ="values")
        elif solver == "anal":
            c_p, _ = self.integ_dCp_analytical(tsr, theta_p, r_min, r_max)
        else:
            raise ValueError(f"Solver must either be 'num' for numerical "
                             "solving or 'anal' for analytical solving, not "
                             f"{solver}")
        
        inv_c_p = 1 - c_p
        
        if inv_c_p <0:
            return 1 
        elif inv_c_p >1:
            return 1 
        else:
            return inv_c_p
    
    def integ_dCp_numerical_madsen (self, tsr, theta_p, r_range, 
                             r_range_type = "values"):
        return self.integ_dCp_numerical (tsr=tsr, theta_p=theta_p, 
                                         r_range=r_range, 
                                         r_range_type = r_range_type, 
                                         gaulert_method = "Madsen")
    
    def calc_cp (self, tsr_range, theta_p_range, 
                      r_range, r_range_type="values",
                      gaulert_method = "classic",
                      multiprocessing = True):
        """Optimization of the tip speed ratio and pitch angle for the 
        maximization of the power coefficient.
        
        Parameters:
            tsr_range (array-like):
                Tip speed ratio range to consider
            theta_p_range (array-like):
                Pitch angle range to consider [deg]
                NOTE: Value is assumed to be specified in degrees
            
            
        Returns:
        
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
        
        ds_cp = xr.Dataset(
            {},
            coords={"tsr":tsr_range,
                    "theta_p":theta_p_range}
            )
        
        ds_cp_shape = [len(tsr_range), len(theta_p_range)]
        
        ds_cp["cp_anal"] = (list(ds_cp.coords.keys()), 
                                 np.empty(ds_cp_shape))
        ds_cp["cp_num"] = (list(ds_cp.coords.keys()), 
                                np.empty(ds_cp_shape))
        ds_cp["cT_num"] = (list(ds_cp.coords.keys()), 
                                np.empty(ds_cp_shape))
        
        if multiprocessing:
            #Multiprocessing with executor.map
            with concurrent.futures.ProcessPoolExecutor() as executor:
                theta_p_comb, tsr_comb = np.meshgrid(theta_p_range, tsr_range)
                theta_p_comb = theta_p_comb.flatten()
                tsr_comb = tsr_comb.flatten()
                
                comb_len = len(tsr_comb)
                
                # integrator_anal = list(executor.map(self.integ_dCp_analytical,
                #                             tsr_comb,
                #                             np.deg2rad(theta_p_comb),
                #                             np.full(comb_len, r_range[0]),
                #                             np.full(comb_len, r_range[-1]),
                #                             np.full(comb_len, gaulert_method)
                #                             ))
                
                if gaulert_method == "classic":
                    integrator_num = list(executor.map(self.integ_dCp_numerical,
                                                tsr_comb,
                                                np.deg2rad(theta_p_comb),
                                                [r_range] * comb_len,
                                                np.full(comb_len, "values")))
                else:
                    integrator_num = list(executor.map(self.integ_dCp_numerical_madsen,
                                                tsr_comb,
                                                np.deg2rad(theta_p_comb),
                                                [r_range] * comb_len,
                                                np.full(comb_len, "values")))
                
                for i in range(comb_len):
                    ds_cp["cp_num"].loc[dict(tsr=tsr_comb[i],
                                           theta_p=theta_p_comb[i])
                                      ] = integrator_num[i][0]
                    ds_cp["cT_num"].loc[dict(tsr=tsr_comb[i],
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
                    
                    # ds_cp["cp_anal"].loc[dict(tsr=tsr_comb[i],
                    #                         theta_p=theta_p_comb[i])
                    #                    ] = integrator_anal[i][0]
        else:
            #No Multiprocessing, just Iterating
            for tsr in tsr_range:
                for theta_p in theta_p_range:
                    # cp_anal, cp_abserr = self.integ_dCp_analytical (
                    #     tsr, np.deg2rad(theta_p), 
                    #     r_min=r_range[0],  r_max=r_range[-1],
                    #     gaulert_method=gaulert_method)
                    
                    cp_num, cT_num, a, a_p, F = self.integ_dCp_numerical (
                        tsr=tsr, theta_p=np.deg2rad(theta_p), r_range=r_range, 
                        r_range_type="values", gaulert_method=gaulert_method)
                    
                    #Save results to dataframe
                    ds_bem["a"].loc[dict(tsr=tsr,theta_p=theta_p)] = a
                    ds_bem["a_p"].loc[dict(tsr=tsr,theta_p=theta_p)] = a_p
                    ds_bem["F"].loc[dict(tsr=tsr,theta_p=theta_p)] = F
                   
                    # ds_cp["cp_anal"].loc[dict(tsr=tsr,theta_p=theta_p)] = cp_anal
                    ds_cp["cp_num"].loc[dict(tsr=tsr,theta_p=theta_p)] = cp_num
                    ds_cp["cT_num"].loc[dict(tsr=tsr,theta_p=theta_p)] = cT_num
        
        return ds_cp, ds_bem
        
    def optimize_C_p (self, tsr_bounds, theta_p_bounds, 
                      r_range, r_range_type="bounds",
                      gaulert_method = "classic"):
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
            
       
        if np.isscalar(tsr_bounds) or not len(tsr_bounds)==2:
            raise TypeError("Bounds for tip speed ratio must be an array-like "
                            + "object with two elements for the lower and upper"
                            + " bound respectively")
        if np.isscalar(theta_p_bounds) or not len(theta_p_bounds)==2:
            raise TypeError("Bounds for pitch angle must be an array-like "
                            + "object with two elements for the lower and upper"
                            + " bound respectively")
        
        
        
        res = scipy.optimize.minimize(self.inv_c_p, 
                                      x0 = (tsr_bounds[0], theta_p_bounds[0]), 
                                      args = (r_range, "values",
                                              gaulert_method, 
                                              "num"),
                                      bounds = [tsr_bounds, theta_p_bounds])
        return res
       
#%% Main    
if __name__ == "__main__":
    R = 89.17
    B = 3 
    P_rated = 10.64*1e6
    v_in = 4
    v_out = 25
    rho = 1.225
    
# ####################################
#     R=31
#     tsr = 2.61*R/8
# ####################################
    
    BEM_calculator =  BEM (R = R,
                           B = B,
                           P_rated = P_rated,
                           v_in = v_in,
                           v_out = v_out,
                           rho = rho)
    

    #Test for BEM Exercise
    # a, a_p, F, conv_res, n = BEM_calculator.converge_BEM(r=24.5, 
    #                                         tsr=tsr, 
    #                                         theta_p = np.deg2rad(-3), 
    #                                         a_0 = 0, 
    #                                         a_p_0 = 0, 
    #                                         epsilon=1e-6, 
    #                                         f = .1, 
    #                                         gaulert_method = "Madsen")
    
    
    # r = 24.5
    # a, a_p, F, res, n = BEM_calculator.converge_BEM(r=r, theta_p=-3,  f = .1, 
    #                                                 gaulert_method = "classic")
    # p_N, p_T = BEM_calculator.calc_local_forces (r=r, theta_p=-3, tsr = tsr, 
    #                                              a=a, a_p=a_p)
    

    # #Plot power coefficent curve
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(ds_cp.coords["tsr"].values, 
    #         ds_cp["cp_num"].sel(theta_p=theta_p[0]).values) 
    # ax.grid()
    # plt.savefig(fname="integ.svg",
    #             bbox_inches = "tight")
    
    
    
    #%% Task 1
    
    #Calculate C_p values
    tsr_l = 5
    tsr_u = 10
    dtsr = .5
    
    theta_p_l = -3
    theta_p_u = 4 
    dtheta_p = .5
    
    r_range = BEM_calculator.bld_df.r
    tsr = np.arange(tsr_l, tsr_u + dtsr, dtsr)
    theta_p=np.arange(theta_p_l, theta_p_u + dtheta_p , dtheta_p)
    
    start = perf_counter()
    ds_cp, ds_bem = BEM_calculator.calc_cp (tsr_range=tsr, 
                                            theta_p_range=theta_p,
                                            r_range=r_range,
                                            r_range_type = "values",
                                            gaulert_method="classic",
                                            multiprocessing=True)
    end = perf_counter()
    print (f"Calculation took {end-start} s")
    
    #Maximum C_P and corresponding coordinates
    c_P_arr= ds_cp["cp_num"]
    c_P_max = c_P_arr.max().item()
    # Find the coordinates where the maximum value occurs
    max_coords = c_P_arr.where(c_P_arr==c_P_arr.max(), drop=True).coords
    # Convert the coordinates to a dictionary for easy access
    coord_dict = {dim: round(coord.values.item(),3) for dim, coord in max_coords.items()}
    tsr_max, theta_p_max = coord_dict.values()
    
    #Plot C_p
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\theta_p$')
    ax.set_zlabel(r'$C_p$')
    X, Y = np.meshgrid(tsr, theta_p)
    Z = ds_cp["cp_num"].values.T
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap="plasma", edgecolor='none')
    ax.set_xticks(np.arange(tsr_l, tsr_u + dtsr, 1))
    ax.set_yticks(np.arange(theta_p_l, theta_p_u + dtheta_p , 1))
    # ax.set_title(r'$C_p$ over $\lambda$ and $\theta_p$')
    plt.savefig(fname="C_P_max_surface_plot.svg")
    
    
    #Plot C_T
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\theta_p$')
    ax.set_zlabel(r'$C_T$')
    X, Y = np.meshgrid(tsr, theta_p)
    Z = ds_cp["cT_num"].values.T
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap="plasma", edgecolor='none')
    ax.set_xticks(np.arange(tsr_l, tsr_u + dtsr, 1))
    ax.set_yticks(np.arange(theta_p_l, theta_p_u + dtheta_p , 1))
    # ax.set_title(r'$C_T$ over $\lambda$ and $\theta_p$')
    plt.savefig(fname="C_T_surface_plot.svg")


#%% Task 2
    
    #Calculate rated wind speed
    V_0 = np.arange(v_in, 12, .25)
    P = c_P_max*.5*rho*np.pi*R**2*V_0**3
    omega = tsr_max*V_0/R
    V_rated = intersection(V_0, P, [v_in, v_out], [P_rated, P_rated])[0][0]
    omega_max = tsr_max*V_rated/R
    rpm_max = omega_max * 60 / (2*np.pi)
    
    # Plot Power over wind speed
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(V_0, P/1e6, label = "Power curve")
    ax.axvline(V_rated, ls="--", lw=1.5, color="k")
    # ax.text(0.2, P_rated/1e6*1.03, '$P_{rated}$', color='k', va='center', ha='center',
    #     transform=ax.get_yaxis_transform())
    ax.axhline(P_rated/1e6, ls="--", lw=1.5, color="k")
    # ax.text(V_rated*.98, .2, '$V_{rated}$', color='k', va='center', ha='center',
    #     transform=ax.get_xaxis_transform(), rotation="vertical")
    # ax.set_title('Power curve')
    ax.set_xlabel('$V_0\:[m/s]$')
    ax.set_ylabel('$P\:[MW]$')
    ax.grid()
    
    plt.savefig(fname="Power_curve.svg",
                bbox_inches = "tight")
    
    
    # Plot omega over wind speed
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(V_0, omega, label = "Power curve")
    ax.axvline(V_rated, ls="--", lw=1.5, color="k")
    # ax.set_title('$\omega$ over the wind speed')
    ax.set_xlabel('$V_0\:[m/s]$')
    ax.set_ylabel('$\omega\:[{rad}/s]$')
    ax.grid()
    
    plt.savefig(fname="omega_over_V0.svg",
                bbox_inches = "tight")

#%% Task 3











