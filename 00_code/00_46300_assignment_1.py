#%% Module imports
#General imports
import scipy
import numpy as np
import pandas as pd
import xarray as xr


#File handling imports
import os
from pathlib import Path

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt

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
mpl.rcParams['text.usetex'] = True          #Use standard latex font

#%% BEM Calculator
class BEM (Utils_BEM):
    def __init__(self, R, P, V_0, tsr, beta, c, C_l, C_d, 
                 v_in = 4, v_out = 25, rho=1.225, B=3,
                 airfoil_files=[], bld_file="", t_airfoils = []):
        self.R = R          #[m] - Rotor radius 
        self.P = P          #[MW] - Rated power
        self.V_0 = V_0      #[m/s] - Free flow velocity
        self.tsr = tsr      #[-] - tip speed ratio
        self.beta = beta    #[-] - Twist of the blade
        self.c = c          #[m] - Chord of the blade
        self.C_l = C_l      #[N/m] - Lift coefficient of the blades
        self.C_d = C_d      #[N/m] - Drag coefficient of the blades
        self.rho = rho      #[kg/m^3] - Air density
        self.B = B          #[-] - Number of blades
        
        #Load the turbine data (incl. checking the inputs for the files)
        # Load the airfoil data (Lift, Drag and Momentum coefficient)
        std_airfoil_files = ['./00_rsc/FFA-W3-2411.txt',
                             './00_rsc/FFA-W3-301.txt',
                             './00_rsc/FFA-W3-360.txt',
                             './00_rsc/FFA-W3-480.txt',
                             './00_rsc/FFA-W3-600.txt',
                             './00_rsc/cylinder.txt']
        
        airfoil_files =  airfoil_files if airfoil_files else std_airfoil_files
        for file in airfoil_files:
            if not os.path.isfile(Path(file)): 
                raise OSError(f"Airfoil file {file} not found")
        self.airfoil_ds, self.airfoil_names = \
            self.load_airfoils_as_xr (airfoil_files)
        
        # Load the blade design data
        bld_file =  bld_file if bld_file else "./00_rsc/bladedat.txt"
        if not os.path.isfile(Path(bld_file)): 
            raise OSError(f"Blade data file {bld_file} not found")
        self.bld_df = pd.DataFrame(columns = ["r", "c", "beta", "tcr"],
                                   data=np.loadtxt(bld_file, skiprows=0))
        
        #Check input for the thickness of the airfoils
        self.airfoil_names = self.airfoil_ds.coords["airfoil"].values
        if (not type(t_airfoils) in [np.ndarray, list, tuple] 
                or not len(t_airfoils)==len(std_airfoil_files))  \
            or (type(t_airfoils) == dict 
                and not set(t_airfoils.keys) == set(self.airfoil_names)):
                t_airfoils = [24.1, 30.1, 36, 48, 60, 100]

        self.t_airfoils = dict(zip(self.airfoil_names, t_airfoils)) 
        
    
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
                Radii at which to calculate the values
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the Turbine
            V_0 (scalar numerical value or array-like):
                Free stream velocity in front of the turbine
                Note: if -1 is used, then the value from the class parameter is
                used
            omega (scalar numerical value or array-like):
                Rotational velocity of the turbine
                Note: if -1 is used, then this parameter is calculated from 
                the tip speed ratio tsr, the free stream velocit V_0 and the 
                rotor radius R
            
        Returns:
            phi (np.float or np.ndarray):
               the angle between the plane of rotation and the relative 
               velocity 
        """
        
        #Check the input values
        a, a_p, r, tsr, V_0, omega = self.check_dims (a, a_p, r, tsr, V_0, omega)
        
        if np.any(a>=1) or np.any(a<0):
            raise ValueError("Axial induction factor must be lie within [0,1[")
        
        if np.any(a_p<0):
            raise ValueError("Tangential induction factor must be lie within "
                             + f"[0,inf[")
        
        if not np.any(tsr<=0):
            phi =  np.arctan((np.divide(1-a, 
                                        (1+a_p)*tsr) * self.R/r).astype(float))
        elif not np.any(V_0<=0) and not np.any(omega<=0):
            phi =  np.arctan(np.divide((1-a)*V_0, 
                                       (1+a_p)*omega*r).astype(float))
        else:
            raise ValueError("Invalid input parameters. Either tsr>0 or "
                             f"V_0>0 and omega>0 need to be given")
        
        return phi
    
    def calc_solidity (self, r):
        """Calculates the solidity of a blade at a certain radius
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the solidity
        
        Returns:
            sigma (scalar numerical value or array-like):
                Blade solidity at radius r
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
        #Check inputs
        # Check if the dimensions of the input values match (must be either 
        # scalar values or array-like of equal shape)
        x_tmp, x_0, f = self.check_dims (x_tmp, x_0, f)
        
        #Relax value
        x = f*x_tmp + (1-f)*x_0 
        return x
    
    def calc_ind_factors(self, r, theta_p = np.pi, 
                         a_0 = 0, a_p_0 = 0,
                         gaulert_method = "classic"):
        """Calulation of the induced velocity factors from the equations of the
        blade element momentum theory.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades
            a_0 (scalar numerical value or array-like):
                Start value for the axial induction factor a (default: 0)
            a_p_0  (scalar numerical value or array-like):
                Start value for the tangential induction factor a' (default: 0)
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
        
        #Check inputs:
        if not type(gaulert_method) == str or gaulert_method not in ['classic', "Madsen"]:
            raise ValueError("Method must be either 'Gaulert' or 'Madsen'")
        
        if r>=.98*self.R:
            return -1, -1, -1
        
        #Calculate the angle of attack
        phi = self.arctan_phi (a_0, a_p_0, r, self.tsr)
        theta = theta_p + self.beta
        
        aoa = phi-theta

        #calculate normal and tangential force factors
        C_n = self.C_l*np.cos(phi) + self.C_d*np.sin(phi)
        C_t = self.C_l*np.sin(phi) - self.C_d*np.cos(phi)
        
        #Calculate solidity
        sigma = np.divide(self.c*self.B, 2*np.pi*r)
        
        #Calculate Prandtl correction factor
        f = self.B/2 * (self.R-r)/(r*np.sin(phi)) 
        F = 2/np.pi * np.arccos(np.exp(-f))
        
        #Calculate induction factors
        
        #Thrust coefficient (Eq. 6.40)
        C_T = np.divide(np.power(1-a_0, 2)*C_n*sigma,
                        np.power(np.sin(phi),2))
        
        if gaulert_method == "classic":
            # Temporary axial induction factor
            if a_0 <.33:
                a = C_T * np.divide (1,
                                     4*F*(1-a_0))
                
                # Full formula in one line (C_T inserted)
                # a_tmp = (sigma*C_n)/(4*F*np.power(np.sin(phi),2)) * (1-a_0)
            else:
                a = C_T * np.divide (1,
                                     4*F*(1-.25 * (5-3*a_0) * a_0))
            
            # Temporary tangential induction factor
            a_p = (1+a_p_0) * np.divide(sigma*C_t,
                                        4*F*np.sin(phi)*np.cos(phi)) 
         
        else:
            a = .246*C_T + .0586*np.power(C_T,2) + .0883*np.power(C_T,3)
            
            #Tangential induction factor without corrections (Eq. 6.36):
            a_p = 1 / (np.divide(4*F*np.sin(phi)*np.cos(phi), sigma*C_t) 
                       - 1)   
        
        return a, a_p, F
    
    def converge_BEM(self, r, theta_p = np.pi, a_0 = 0, a_p_0 = 0, 
                     epsilon=1e-6, f = .1, gaulert_method = "classic"):
        """Iterative solver of the equations of blade element momentum theory 
        for the induced velocity factors.
        Note: If the values do not converge within 1000 iterations, the 
        calculation is stopped
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades
            a_0 (scalar numerical value or array-like):
                Start value for the axial induction factor a (default: 0)
            a_p_0  (scalar numerical value or array-like):
                Start value for the tangential induction factor a' (default: 0)
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
        
        a, a_p, F = self.calc_ind_factors(r, theta_p, a_0, a_p_0, 
                                          gaulert_method=gaulert_method)
        n = 0
        
        while (abs(a-a_0)>epsilon) or (abs(a_p-a_p_0)>epsilon): 
            if n>=1000:
                print(f"Maximum iteration number reached before convergence")
                break
            a_0, a_p_0 = a, a_p
            a_tmp, a_p_tmp, F = self.calc_ind_factors(r, theta_p, a_0, a_p_0, 
                                                      gaulert_method=gaulert_method)
            a = self.relax_parameter(a_tmp, a_0, f)
            a_p = self.relax_parameter(a_p_tmp, a_p_0, f)

            n +=1
        
        # if n<=1000:
        #     print(f"Calculation stopped after {n} iteration")
        
        conv_res = (abs(a-a_0), abs(a_p-a_p_0))
        
        return a, a_p, F, conv_res, n
    
    def calc_local_forces (self, r, tsr, a=-1, a_p=-1, theta_p=np.pi):
        """Calculates the local forces for given turbine parameters.
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            a (scalar numerical value or array-like):
                Axial induction factor a.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            a_p (scalar numerical value or array-like):
                Tangential induction factor a'.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades
            
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
            a, a_p, F, _, _ = self.converge_BEM(r, theta_p)
        
        #Check if there are any zero values in a, a_p and tsr (not allowed)
        if not all(np.all(var) for var in [a, a_p, tsr]):
            raise ValueError("No zero values allowed in a, a* and tsr")

        #Calculate the angle of attack
        phi = self.arctan_phi (a, a_p, r, tsr=self.tsr)
        theta = theta_p + self.beta
        
        aoa = phi-theta
        
        #Calculate relative velocity
        omega = tsr*self.V_0/self.R
        V_rel = np.sqrt((np.power(self.V_0*(1-a),2) 
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
    
    def calc_local_tang_force (self, r, tsr, a=-1, a_p=-1, theta_p=np.pi):
        """Calculates the local tangential force for given turbine parameters.
        All turbine parameters, which are not part of the inputs, are read from
        the class attributes.
        If the calculation of the local tangential force produces an error or 
        returns an invalid value, 0 is returned
        
        Parameters:
            r (scalar numerical value or array-like):
                Radii at which to calculate the values
            tsr (scalar numerical value or array-like):
                Tip speed ratio of the turbine 
            a (scalar numerical value or array-like):
                Axial induction factor a.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            a_p (scalar numerical value or array-like):
                Tangential induction factor a'.
                If -1 is used for this parameter, then the axial and tangential
                induction factors are calculated from the BEM method
            theta_p (scalar numerical value or array-like):
                Pitch angle of the blades
            
        Returns:
            p_T (np.float or np.ndarray): 
                Local tangential force in N/m
        """
        
        p_N, p_T = self.calc_local_forces (r, tsr, a=-1, a_p=-1, theta_p=np.pi)
        
        return p_T
    
    def optimize_C_p (self, tsr_range, theta_p_range, r_range = []):
        """Optimization of the tip speed ratio and pitch angle for the 
        maximization of the power coefficient.
        
        Parameters:
            tsr_range (array-like):
                Tip speed ratio range to consider
            theta_p_range (array-like):
                Pitch angle range to consider
            
            
        Returns:
        
        """
        
        #Prepare inputs
        if not r_range: r_range = np.arange(1,self.R,.1)
        tsr_range = np.array(tsr_range)
        theta_p_range = np.array(theta_p_range)
        
        #Prepare dataset for the values
        bem_ds = xr.Dataset(
            {},
            coords={"r":r_range, 
                    "tsr":tsr_range,
                    "theta_p":theta_p_range}
            )
        
        ds_bem_shape = [len(r_range), len(tsr_range), len(theta_p_range)]
        
        bem_ds["a"] = (list(bem_ds.coords.keys()),
                       np.empty(ds_bem_shape))
        bem_ds["a_p"] = (list(bem_ds.coords.keys()),
                         np.empty(ds_bem_shape))
        bem_ds["F"] = (list(bem_ds.coords.keys()),
                       np.empty(ds_bem_shape))
        bem_ds["p_T"] = (list(bem_ds.coords.keys()), 
                         np.empty(ds_bem_shape))
        
        #Prepare dataset for the values
        M_ds = xr.Dataset(
            {},
            coords={"tsr":tsr_range,
                    "theta_p":theta_p_range}
            )
        
        ds_m_shape = [len(tsr_range), len(theta_p_range)]
        
        M_ds["M_int"] = (list(M_ds.coords.keys()), 
                                 np.empty(ds_m_shape))
        M_ds["M_iter"] = (list(M_ds.coords.keys()), 
                                np.empty(ds_m_shape))
        
# =============================================================================
#         #Plot power coefficent curve
#         fig, ax = plt.subplots(figsize=(16, 10))
#         ax.plot(r_range, p_T)
#         ax.grid()
#         plt.savefig(fname="integ.svg",
#                     bbox_inches = "tight")
# =============================================================================
        
        for tsr in tsr_range:
            for theta_p in theta_p_range:
                #Integrate p_T over r as a function
                M_integ, M_abserr = scipy.integrate.quad(self.calc_local_tang_force, 
                                        1, self.R-1,
                                        args = (tsr, -1, -1, theta_p))
                
                #Integrate p_T over r using discrete samples for r
                for r in r_range:
                    print (f"r = {r}")
                    a, a_p, F, _, _ = self.converge_BEM(r, theta_p)
                    bem_ds["a"].loc[dict(r=r,tsr=tsr,theta_p=theta_p)] = a
                    bem_ds["a_p"].loc[dict(r=r,tsr=tsr,theta_p=theta_p)] = a_p
                    bem_ds["F"].loc[dict(r=r,tsr=tsr,theta_p=theta_p)] = F
                
                p_N, p_T = self.calc_local_forces (
                    r=r_range, 
                    tsr=tsr,
                    a=bem_ds.sel(tsr=tsr,theta_p=theta_p)["a"].values,
                    a_p=bem_ds.sel(tsr=tsr,theta_p=theta_p)["a_p"].values, 
                    theta_p=theta_p
                    )
                
                M_iter = scipy.integrate.trapezoid(p_T, r_range)
                
                #Save results to dataframe
                M_ds["M_int"].loc[dict(tsr=tsr,theta_p=theta_p)] = M_integ
                M_ds["M_iter"].loc[dict(tsr=tsr,theta_p=theta_p)] = M_iter
                
        return M_ds, bem_ds
                
       
#%% Main    
if __name__ == "__main__":
    omega = 2.61
    R = 31
    V_0 = 8
    tsr = omega*R/V_0
    
    BEM_calculator =  BEM (R = R,
                           P = 10,
                           V_0 = V_0, 
                           tsr = tsr, 
                           beta = 2, 
                           c = 1.5, 
                           C_l = .5, 
                           C_d = .01, 
                           rho=1.225, 
                           B=3)
    	
    r = 24.5
    a, a_p, F, res, n = BEM_calculator.converge_BEM(r=r, theta_p=-3, 
                                            f = .1,
                                            gaulert_method = "classic")
    p_N, p_T = BEM_calculator.calc_local_forces (r=r, theta_p=-3, tsr = tsr, 
                                                 a=a, a_p=a_p)
    
    
    # tsr = np.arange(5,10)
    # theta_p=np.arange(-3,4)
    # M_ds, bem_ds = BEM_calculator.optimize_C_p (tsr_range=tsr, theta_p_range=[-3])
    
    
