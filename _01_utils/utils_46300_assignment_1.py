import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from pathlib import Path

class Utils_BEM():
    def __init__(self, airfoil_files=[], bld_file="", t_airfoils = []):
        #Load the turbine data (incl. checking the inputs for the files)
        # Load the airfoil data (Lift, Drag and Momentum coefficient)
        
        source_folder = r"C:\Users\davis\00_data\00_Documents\01_Master_studies"\
                        + r"\46300 - WTT and Aerodynamics\01_Assigments"\
                        + r"\00_Assignment_1\00_code\_00_rsc"
        std_airfoil_files = [r'FFA-W3-2411.txt',
                             r'FFA-W3-301.txt',
                             r'FFA-W3-360.txt',
                             r'FFA-W3-480.txt',
                             r'FFA-W3-600.txt',
                             r'cylinder.txt']
        std_airfoil_files = [source_folder + "\\" + s for s in std_airfoil_files]
        
        airfoil_files =  airfoil_files if airfoil_files else std_airfoil_files
        for file in airfoil_files:
            if not os.path.isfile(Path(file)): 
                raise OSError(f"Airfoil file {file} not found")
        self.airfoil_ds, self.airfoil_names = \
            self.load_airfoils_as_xr (airfoil_files)
        
        # Load the blade design data
        bld_file =  bld_file if bld_file else source_folder + "\\" + "bladedat.txt"
        if not os.path.isfile(Path(bld_file)): 
            raise OSError(f"Blade data file {bld_file} not found")
        self.bld_df = pd.DataFrame(columns = ["r", "beta", "c", "tcr"],
                                   data=np.loadtxt(bld_file, skiprows=0))
        
        #Check input for the thickness of the airfoils
        self.airfoil_names = self.airfoil_ds.coords["airfoil"].values
        if (not type(t_airfoils) in [np.ndarray, list, tuple] 
                or not len(t_airfoils)==len(std_airfoil_files))  \
            or (type(t_airfoils) == dict 
                and not set(t_airfoils.keys) == set(self.airfoil_names)):
                t_airfoils = [24.1, 30.1, 36, 48, 60, 100]

        self.t_airfoils = dict(zip(self.airfoil_names, t_airfoils)) 
        
        #Create arrays from the datasets for faster access
        self.aoa_arr = self.airfoil_ds.coords["aoa"].values
        self.cl_arr = self.airfoil_ds["c_l"].values
        self.cd_arr = self.airfoil_ds["c_d"].values

    @staticmethod
    def check_shape(var):
        """Checks the shape of a variable and converts it to a numpy array, if
        it is a list
        
        Parameters:
            var (scalar number or array-like): 
                the variable for which to check the state
        
        Return:
            var (scalar number or numpy array):
                the variable
            shape (int or tuple):
                the shape of the variable
        """
        
        var_type = type (var)
        
        if type (var) == list: 
            var = np.array (var)
            shape = var.shape
        elif type(var) == np.ndarray:
            shape = var.shape
        elif type(var) in [int, float, 
                           np.int16, np.int32, np.int64, 
                           np.float16, np.float32, np.float64]:
            var = np.float64(var)
            shape = 1
        else:
            shape = -1
            
        return var, shape
    
    def check_dims (self, *args):
        """Checks the shape of the variables in *args and if there are 
        non-scalar variables, checks if they all have the same shape.
        In addition, all non-scalar variables are converted to numpy arrays
        
        Parameters:
            args:
                The variables to check
            
        Returns:
            var_lst (list):
                The variables from *args
        """
        #Convert the arguments to numpy arrays if they are lists and find out
        # their respective shapes
        var_lst, dims = [], np.array([])
        for var in args:
            var, dim = self.check_shape(var)
            var_lst.append(var)
            dims = np.append(dims, dim)
        
        #Check if the variables, which are arrays, have the same shape (if not
        # raise an error)
        arr_dims = dims[dims!=1]
        if len(set(arr_dims.flatten()))>1:
            raise ValueError ("Dimensions of the input arrays must match")
        
        return var_lst   
    
    def load_airfoils_as_xr (self, file_lst):
        """Loads the lift, drag and momentum coefficient (C_l, C_d & C_m)
        from a list of txt files and returns them as a xarray Dataset.
        It is assumed, that the first column in the txt file are the angles
        of attack (aoa) and that these are the same for all files. The 
        residual columns are assumed to be the lift, drag and momentum 
        coefficient (in this order)
        
        Parameters:
            file_lst (list):
                A list containing the relative or absolute file paths of 
                the text files
        
        Returns:
            airfoil_data (xarray Dataset):
                Dataset with the C_l, C_d and C_m values (as separate 
                coordinates) for each airfoil. The coordinates are the 
                angles of attack
            file_names (list):
                The names of the files which were read
        """
        #Initializing tables 
        aoa_arr,_,_,_ = np.loadtxt(file_lst[0], skiprows=0).T
        file_names = [Path(file).stem for file in file_lst]
        ds_shape = (len(aoa_arr), len(file_lst))
        
        airfoil_data = xr.Dataset(
            dict(
                c_l=(["aoa", "airfoil"], np.empty(ds_shape)),
                c_d=(["aoa", "airfoil"], np.empty(ds_shape)),
                c_m=(["aoa", "airfoil"], np.empty(ds_shape))
                ),
            coords={"aoa":aoa_arr,
                    "airfoil": file_names}
            )
        
        #Readin of tables. Only do this once at startup of simulation
        for name, fpath in zip(file_names, file_lst):
            airfoil_data["c_l"].loc[{"airfoil":name}], \
            airfoil_data["c_d"].loc[{"airfoil":name}], \
            airfoil_data["c_m"].loc[{"airfoil":name}] = \
                np.loadtxt(fpath, skiprows=0, usecols=[1,2,3]).T

        return airfoil_data, file_names
    
    def interp_coeffs (self, aoa, tcr):
        """Interpolates the lift and drag coefficient from the look-up tables 
        in the airfoil_ds.
        Double interpolation is necessary to first interpolate the C_l & C_d
        values for each airfoil given an angle of attack aoa. The second 
        interpolation then interpolates between the airfoils based on their
        thickness.
        
        Parameters:
            aoa (scalar numerical value or array-like):
                Angles of attack [deg]
            tcr (scalar numerical value or array-like):
                Thickness to chord length ratios of the airfoil [m]
              
        Returns:
            C_l (scalar numerical value or array-like):
                Lift coefficients 
            C_d (scalar numerical value or array-like):
                Drag coefficients 
        """

        #Determine lift and drag coefficients
        aoa = np.array(aoa).reshape(-1)
        tcr = np.array(tcr).reshape(-1)
        
        cl_aoa=np.zeros([aoa.size,6])
        cd_aoa=np.zeros([aoa.size,6])
        
        for i in range(6):
            cl_aoa[:,i]=np.interp (aoa,
                                   self.aoa_arr,
                                   self.cl_arr[:,i])
            cd_aoa[:,i]=np.interp (aoa,
                                   self.aoa_arr,
                                   self.cd_arr[:,i])
        
        C_l = np.array([np.interp(tcr[i], [*self.t_airfoils.values()], cl_aoa[i,:]) 
                        for i in range(tcr.size)])
        C_d = np.array([np.interp(tcr[i], [*self.t_airfoils.values()], cd_aoa[i,:]) 
                        for i in range(tcr.size)])
        
        return C_l, C_d
    
    def check_radius_range (self, r_range, R):
        """Checks inputs for a range of radii for completeness and 
        reasonability.
        
        Parameters:
            r_range (array-like):
                A list containing the minimum & maximum radius and the step 
                width. 
                Alternatively, the default values of r_min=0, r_max=R and dr=.5
                can be used
            R (int or float):
                The rotor radius
                
        Returns:
            r_range (numpy array):
                An array with values from r_min to r_max with step width dr
            r_min (float):
                lower boundary of the radius range
            r_max (float):
                upper boundary of the radius range
            dr (float):
                Step width of the radius
        """
        
        #Check inputs for completeness
        if len(r_range) == 0:
            r_min = 1
            r_max = R
            dr = .5
        elif len(r_range) == 1:
            r_min = r_range[0]
            r_max = R
            dr = .5
        elif len(r_range) == 2:
            r_min, r_max = r_range
            dr = .5
        elif len(r_range) == 3:
            r_min, r_max, dr = r_range
        else:
            raise ValueError("For radius range type 'bounds', the r_range "
                             + "parameter needs to contain the bounds and "
                             + "the step in the form [r_min, r_max, dr]")
        
        if r_min< 0 or r_min>R:
            raise ValueError("Lower bound must be within [0,R]")
        
        if r_min>r_max:
            raise ValueError("Upper bound must be higher than lower bound")
        elif r_max>R:
            raise ValueError("Upper bound must be within [0,R]")
            
        if dr< 0:
            raise ValueError("Step width must be positive")
        elif dr>r_max-r_min:
            raise ValueError("Step width must be smaller than interval of"
                             + "r_min and r_max")
    
        r_range = np.arange(r_min, r_max+dr/2, dr)
        
        return r_range, r_min, r_max, dr
        