# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:56:42 2022

@author: cgrinde
"""
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
#TEST OF INTERPOLATION ROUTINE. COMPARE TO INTERP1 IN MATLAB


files=['FFA-W3-2411.txt','FFA-W3-301.txt','FFA-W3-360.txt','FFA-W3-480.txt','FFA-W3-600.txt','cylinder.txt']
#Initializing tables    
cl_tab=np.zeros([105,6])
cd_tab=np.zeros([105,6])
cm_tab=np.zeros([105,6])
aoa_tab=np.zeros([105,])
#Readin of tables. Only do this once at startup of simulation
for i in range(np.size(files)):
     aoa_tab[:],cl_tab[:,i],cd_tab[:,i],cm_tab[:,i] = np.loadtxt(files[i], skiprows=0).T


def load_airfoils_as_xr (file_lst):
    """Loads the lift, drag and momentum coefficient (C_l, C_d & C_m)
    from a list of txt files.
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
    """
    #Initializing tables 
    aoa_arr,_,_,_ = np.loadtxt(file_lst[0], skiprows=0).T
    file_names = [Path(file).stem for file in file_lst]
    ds_shape = (len(aoa_arr), len(file_lst))
    
    airfoil_data = xr.Dataset(
        dict(
            c_l=(["aoa", "airfoil"], np.full(ds_shape, np.nan)),
            c_d=(["aoa", "airfoil"], np.full(ds_shape, np.nan)),
            c_m=(["aoa", "airfoil"], np.full(ds_shape, np.nan))
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

    return airfoil_data
airfoil_ds =  load_airfoils_as_xr(files)

# Thickness of the airfoils considered
# NOTE THAT IN PYTHON THE INTERPOLATION REQUIRES THAT THE VALUES INCREASE IN THE VECTOR!

thick_prof=np.zeros(6)
thick_prof[0]=24.1;
thick_prof[1]=30.1;
thick_prof[2]=36;
thick_prof[3]=48;
thick_prof[4]=60;
thick_prof[5]=100;





def force_coeffs_10MW(angle_of_attack,thick,aoa_tab,cl_tab,cd_tab,cm_tab):
    cl_aoa=np.zeros([1,6])
    cd_aoa=np.zeros([1,6])
    cm_aoa=np.zeros([1,6])
    

    #Interpolate to current angle of attack:
    for i in range(np.size(files)):
        cl_aoa[0,i]=np.interp (angle_of_attack,aoa_tab,cl_tab[:,i])
        cd_aoa[0,i]=np.interp (angle_of_attack,aoa_tab,cd_tab[:,i])
        cm_aoa[0,i]=np.interp (angle_of_attack,aoa_tab,cm_tab[:,i])
    
    #Interpolate to current thickness:
    cl=np.interp (thick,thick_prof,cl_aoa[0,:])
    cd=np.interp (thick,thick_prof,cd_aoa[0,:])
    cm=np.interp (thick,thick_prof,cm_aoa[0,:])


    return cl, cd, cm 



# # Lets test it:
# angle_of_attack=-10 # in degrees
# thick = 27 # in percent !
# [clift,cdrag,cmom]=force_coeffs_10MW(angle_of_attack,thick,aoa_tab,cl_tab,cd_tab,cm_tab)

# print('cl:',clift)
# print('cd:',cdrag)
# print('cm:',cmom)