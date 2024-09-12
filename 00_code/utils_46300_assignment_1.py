import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

class Utils_BEM():
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
        elif type(var) in [int, float, np.float16, np.float32, np.float64]:
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
    
    def load_airfoils_as_df (self, file_lst):
        """Loads the lift, drag and momentum coefficient (C_l, C_d & C_m)
        from a list of txt files and returns them as pandas Dataframes.
        It is assumed, that the first column in the txt file are the angles
        of attack (aoa) and that these are the same for all files. The 
        residual columns are assumed to be the lift, drag and momentum 
        coefficient (in this order)
        
        Parameters:
            file_lst (list):
                A list containing the relative or absolute file paths of 
                the text files
        
        Returns:
            cl_df (pandas DataFrame):
                Dataframe with the C_l values for each airfoil. Each 
                airfoil is a column with the respective name of the file,
                the data was extracted from. The indices are the angles of
                attack
            cd_df (pandas DataFrame):
                Dataframe with the C_d values for each airfoil. Each 
                airfoil is a column with the respective name of the file,
                the data was extracted from. The indices are the angles of
                attack
            cm_df (pandas DataFrame):
                Dataframe with the C_m values for each airfoil. Each 
                airfoil is a column with the respective name of the file,
                the data was extracted from. The indices are the angles of
                attack
            aoa_arr (numpy array):
                The angles of attack
        """
        #Initializing tables 
        aoa_arr,_,_,_ = np.loadtxt(file_lst[0], skiprows=0).T
        cl_df = pd.DataFrame(dict(aoa=aoa_arr))
        cd_df = pd.DataFrame(dict(aoa=aoa_arr))
        cm_df = pd.DataFrame(dict(aoa=aoa_arr))
        
        #Readin of tables. Only do this once at startup of simulation
        for file in file_lst:
            fname = Path(file).stem
            cl_df[fname],cd_df[fname],cm_df[fname] = \
                np.loadtxt(file, skiprows=0, usecols=[1,2,3]).T
        
        cl_df.set_index("aoa", inplace = True, drop=True)
        cd_df.set_index("aoa", inplace = True, drop=True)
        cm_df.set_index("aoa", inplace = True, drop=True)
        
        return aoa_arr, cl_df, cd_df, cm_df
    
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