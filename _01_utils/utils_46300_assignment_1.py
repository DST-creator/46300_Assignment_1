#%% Imports
#General imports
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d


#File Handling imports
import os
from pathlib import Path

#Concurrency imports
import ctypes

#Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

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
mpl.rcParams['legend.fontsize'] = 25

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
# mpl.rcParams['ztick.major.pad'] = 5
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

#%% Utils Class
class Utils_BEM():
    def __init__(self, airfoil_files=[], bld_file="", t_airfoils = []):
        #Load the turbine data (incl. checking the inputs for the files)
        # Load the airfoil data (Lift, Drag and Momentum coefficient)
        
        source_folder = r"C:\Users\davis\00_data\00_Documents\01_Master_studies"\
                        + r"\46300 - WTT and Aerodynamics\01_Assigments"\
                        + r"\00_code\_00_rsc"
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
    
        r_range = np.arange(r_min, r_max+dr, dr)
        
        return r_range, r_min, r_max, dr
        
    def plot_3d_data(self, X, Y, Z, 
                     xticks=np.array([]), yticks=np.array([]),
                     plt_type="contour", azim=45, elev=30,
                     labels=["x", "y", "z"], unit_z = "-",
                     hline=None, hline_label="",
                     vline=None, vline_label="",
                     intersect_label="",
                     exp_fld = "_03_export", fname ="", 
                     return_obj=False):
        """Plot a variable Z over a meshgrid X & Y as a contour or surface plot.
        
        Parameters:
            X (m x n array):
                X-values as a meshgrid
            Y (m x n array):
                Y-values as a meshgrid
            Z (m x n array):
                Z-values as a meshgrid
            x_ticks (array-like - optional):
                Tick values for the x-axis
            y_ticks (array-like - optional):
                Tick values for the y-axis
            plt_type (str - optional):
                Plot type. Can be either:
                - 'contour': Contour plot with the Z-values as a colormap
                - 'surface': 3d-surface plot
            azim (int or float - optional):
                Azimuth angle for the 3d view projection (only relevant for 
                plt_type 'surface')
            elev (int or float - optional):
                Elevation angle for the 3d view projection (only relevant for 
                plt_type 'surface')
            labels (array-like - optional):
                Labels for the three axes - Must have length 3!
            hline (bool, int or float - optional):
                Position for a hline to be plotted (if None is specified, no
                hline is drawn)
            hline_label (String - optional):
                Labels for the hline
            vline (bool, int or float - optional):
                Position for a vline to be plotted (if None is specified, no
                hline is drawn)
            vline_label (String - optional):
                Labels for the vline
            intersect_label (String - optional):
                Labels for the intersection point of the hline and vline
            exp_fld (str - optional):
                Folder in which to save the figure. Default: _03_export 
                subfolder of current working directory
            fname (str - optional):
                Filename for export of the plot. If none is provided, a 
                filename is chosen based on the z-label and the plot_type
            return_obj (bool - optional):
                Selection whether the figure and axes object should be returned
            
            
        Returns: 
            fig (matplotlib figure object):
                The figure of the plot (only returned if return_obj=True)
            ax (matplotlib figure object):
                The axes of the plot (only returned if return_obj=True)
        """
        
        if not unit_z == "-":
            label_z = labels[2] + "$\:[" + unit_z +"]$"
        else:
            label_z = labels[2]
        
        if plt_type == "surface":
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            # ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap="plasma", edgecolor='none')
            ax.set_zlabel(label_z)
            ax.view_init(elev, azim)
            
            #Adjust padding
            # fig.tight_layout(pad=5)
            ax.set_box_aspect([1, 1, 1.15])
            ax.zaxis.labelpad = 35 
            ax.zaxis.set_tick_params(pad=15)  # Increase distance of z-ticks from axis
            
        elif plt_type == "contour":
            fig,ax = plt.subplots(figsize=(16, 10))
            contour = ax.contourf(X, Y, Z, 80, cmap='plasma')
            plt.colorbar(contour, 
                         label=label_z,
                         ax=ax)

            width, height = self.get_ax_size(fig, ax)
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            
            if type(hline) in [int, float, np.float16, np.float32, np.float64, 
                               np.int16, np.int32, np.int64]: 
                plt_hline = True
            else:
                plt_hline = False
            if type(vline) in [int, float, np.float16, np.float32, np.float64, 
                               np.int16, np.int32, np.int64]: 
                plt_vline = True
            else:
                plt_vline = False
            
            if plt_hline:
                ax.axhline(hline, c="k", ls="--", lw=1.2)
                if hline_label:
                    y_pos = self.calc_text_pos(ax_lims=ylims, ax_size=height, 
                                               base_pos=hline, offset=20)
                    if plt_vline:
                        x_pos = self.calc_text_pos(ax_lims=xlims, ax_size=width, 
                                                   base_pos=vline)
                        ax.text(x_pos, y_pos,  hline_label, 
                                color='k', va='bottom', ha='right', 
                                size = "medium")
                    else:
                        ax.text(0.2, y_pos,  hline_label, 
                                color='k', va='bottom', ha='center', 
                                size = "medium", 
                                transform=ax.get_yaxis_transform())
                plt_hline = True
            if plt_vline:
                ax.axvline(vline, c="k", ls="--", lw=1.2)
                if vline_label:
                    x_pos = self.calc_text_pos(ax_lims=xlims, ax_size=width, 
                                               base_pos=vline, offset=-20)
                    if plt_hline:
                        y_pos = self.calc_text_pos(ax_lims=ylims, 
                                                   ax_size=height, 
                                                   base_pos=hline)
                        ax.text(x_pos, y_pos,  vline_label, 
                                color='k', va='top', ha='right', 
                                size = "medium", rotation="vertical")
                    else:
                        ax.text(x_pos, 0.2,  vline_label, 
                                color='k', va='center', ha='right', 
                                size = "medium", rotation="vertical",
                                transform=ax.get_xaxis_transform())
                    
                
                if plt_hline:
                    ax.scatter(vline, hline, marker ="o", c = "k", s=50)
                    
                    if intersect_label:
                        arrowstyle = dict(arrowstyle="->", 
                                          connectionstyle="angle,angleA=0,angleB=60")
                        ax.annotate(intersect_label, (vline,hline), (60, 40), 
                                     xycoords='data', 
                                     textcoords='offset points', 
                                     ha='left', va='bottom', 
                                     arrowprops = arrowstyle)
                
        else:
            print(f"Unknown plot type {plt_type}")
            return
        
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if np.size(xticks)>1:
            ax.set_xticks(xticks)
        if np.size(yticks)>1:
            ax.set_yticks(yticks)
        
        if not fname: fname= f"{labels[2].replace('$','')}_{plt_type}"

        fig.savefig(fname = Path(exp_fld, fname + ".svg"))
        fig.savefig(fname = Path(exp_fld, fname + ".pdf"), format="pdf")        # Save PDF for inclusion
        fig.savefig(fname = Path(exp_fld, fname + ".pgf"))                      # Save PGF file for text inclusion in LaTeX
        
        if return_obj:
            return fig,ax
        else:
            plt.close(fig)
            
    def test_neg_a(self, r, tsr_range = np.arange(1,10,.5), 
                   theta_p_range=np.arange(0,40,1), 
                   a_0=0, a_p_0 =0):
        """Plot exemplary operation range for a given tip speed ratio and 
        pitch angle range at a specific radius r
        
        Parameters:
            r (int or float):
                Radius to evaluate
            tsr_range (array-like):
                Range of tip speed ratio values
            theta_p_range  (array-like):
                Range of pitch angles [deg]
            a_0 (int or float):
                Axial induction factor
            a_p_0 (int or float):
                tTangential induction factor
            
        Returns:
            None
        """
        
        c = np.interp(r, self.bld_df.r, self.bld_df.c) 
        tcr = np.interp(r, self.bld_df.r, self.bld_df.tcr) 
        beta = np.interp(r, self.bld_df.r, self.bld_df.beta)
        sigma = np.divide(c*self.B, 2*np.pi*r)
        
        phi_range =  np.rad2deg(np.arctan((np.divide(1-a_0, 
                                        (1+a_p_0)*tsr_range) * self.R/r)))
        
        phi_mesh, theta_mesh = np.meshgrid(phi_range, theta_p_range)
        phi_mesh_rad = np.deg2rad(phi_mesh)
        
        
        aoa = phi_mesh - (beta + theta_mesh)
        
        C_l, C_d = np.zeros(aoa.shape), np.zeros(aoa.shape)
        
        for iy, ix in np.ndindex(aoa.shape):
            C_l_i, C_d_i = self.interp_coeffs (aoa=aoa[iy, ix], 
                                           tcr=tcr)
            C_l[iy, ix], C_d[iy, ix] = C_l_i[0], C_d_i[0]
        
        #calculate normal and tangential force factors
        C_n = C_l*np.cos(phi_mesh_rad) + C_d*np.sin(phi_mesh_rad)
        C_t = C_l*np.sin(phi_mesh_rad) - C_d*np.cos(phi_mesh_rad)  
        
        #Calculate Prandtl correction factor
        f_corr = self.B/2 * (self.R-r)/(r*np.sin(phi_mesh_rad)) 
        F = 2/np.pi * np.arccos(np.exp(-f_corr))
        
        #Calculate induction factors
        #Thrust coefficient (Eq. 6.40)
        dC_T = np.divide(np.power(1-a_0, 2)*C_n*sigma,
                        np.power(np.sin(phi_mesh_rad),2))
        
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
                                    4*F*np.sin(phi_mesh_rad)*np.cos(phi_mesh_rad)) 
        
        
        tsr_mesh, _ = np.meshgrid(tsr_range, theta_p_range)
        
        #Plot a
        self.plot_3d_data(X=tsr_mesh, Y=theta_mesh, Z=a, 
                         plt_type="contour", azim=100,
                         labels=[r"$\lambda\:[-]$", 
                                 r"$\theta_p\:[deg]$", 
                                 r"$a$"],
                         exp_fld = "_02_testing",
                         fname = "a_operating_range_contour.svg")
        
        #Set negative a values to 0 and plot again
        a[a<0]=0
        self.plot_3d_data(X=tsr_mesh, Y=theta_mesh, Z=a, 
                         plt_type="surface", azim=100,
                         labels=[r"$\lambda\:[-]$", 
                                 r"$\theta_p\:[deg]$", 
                                 r"$a$"],
                         exp_fld = "_02_testing",
                         fname = "a_operating_range_surface.svg")
        
        
        
        #Plot a_p
        self.plot_3d_data(X=tsr_mesh, Y=theta_mesh, Z=a_p, 
                         plt_type="contour", 
                         labels=[r"$\lambda\:[-]$", 
                                 r"$\theta_p\:[deg]$", 
                                 r"$a_p$"],
                         exp_fld = "_02_testing",
                         fname = "a_p_operating_range_contour.svg")
        
# =============================================================================
#         #Plot as 2d lines
#         fig, ax = plt.subplots(figsize=(16, 10))
#         for z in range(len(theta_p_range)):
#             ax.plot(tsr_range, a[z, :])
#         ax.set_xlabel(r"$\lambda\:[-]$")
#         ax.set_ylabel(r"$a$")
#         plt.grid()
#         
#         plt.savefig(fname="./_03_export/" + "a_survey_2d.svg")
#         plt.close(fig)
# =============================================================================

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
    
    @staticmethod
    def plot_weibull(A=9, k=1.9):
        
        V_range = np.arange(0, 31)
        h_w = k/A*np.power(V_range/A, k-1)*np.exp(-np.power(V_range/A, k))
        h_w_cum = 1- np.exp(-np.power(V_range/A, k))
        
        # Plot probability distribution over wind speed
        fig, ax1 = plt.subplots(figsize=(16, 10))
        plt_dist = ax1.plot(V_range, h_w*100, ls="-", 
                            label = "Weibull distribution")
        ax1.set_xlabel('$V_0\:[m/s]$')
        ax1.set_ylabel('$h_w\:[\%]$')
        ax1.grid()
        
        # Plot cumulative probability over wind speed
        ax2 = ax1.twinx()
        plt_cum = ax2.plot(V_range, h_w_cum*100, ls="--", 
                           label = "Weibull cumulative density function")
        ax2.set_ylabel('$h_{w,cum}\:[\%]$')
        
        
        lns = plt_dist+plt_cum
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        
        fname = "_03_export/Weibull_dist"
        fig.savefig(fname+".svg")
        fig.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
        plt.close(fig)
    
    @staticmethod
    def available_power(V, rho, R):
        return .5*rho*np.pi*(R**2)*np.power(V,3)
    
    @staticmethod
    def get_ax_size(fig, ax):
        """Calculates the axes size in pixels
        
        Parameters:
            fig (matplotlibe figure):
                Figure for which to calculate the axes size 
            ax (matplotlibe axes):
                Axes for which to calculate the axes size 
                
        Returns:
            width (float):
                Width of the figure in pixels
            height (float):
                height of the figure in pixels
        """
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        return width, height
    
    @staticmethod
    def calc_text_pos(ax_lims, ax_size, base_pos, offset=-80):
        """Calculate the position of Text based on an offset in pixels.
        
        Parameters:
            ax_lims (array-like):
                Lower and upper limits of the axis
            ax_size (float):
                Size of the axis in pixels
            base_pos (float):
                Base position of the text in the unit of the axis ticks
            offset (float):
                Desired offset of the text from the base_pos in pixels
        
        Returns:
            pos (float):
                Adjusted position of the text in the unit of the axis ticks
        """
        
        val_len = ax_lims[1]-ax_lims[0]
        pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
        if pixel_pos <100:
            offset = -offset*4
            pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
        elif ax_size-pixel_pos <100:
            offset = -offset*4
            pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
        
        return (pixel_pos)/ax_size*val_len+ ax_lims[0]
        
        