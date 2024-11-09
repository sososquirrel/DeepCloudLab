import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm

from pySAMetrics.Simulation import Simulation

# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation import Simulation
from pySAMetrics.utils import expand_array_to_tzyx_array
from pySAMetrics.utils_3d_functions import get_sequence_images

# Initialize the main dictionary
data_dict = {
    'RCE_T300_U0_B0.1_M1': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '0.1', 'microphysic': '1'},
    'RCE_T300_U0_B0.5_M1': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '0.5', 'microphysic': '1'},
    'RCE_T300_U0_B1_M1': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T300_U10_B1_M1': {'velocity': '10', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T300_U20_B1_M1': {'velocity': '20', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T300_U2.5_B1_M1': {'velocity': '2.5', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T300_U5_B1_M1': {'velocity': '5', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T302_U0_B1_M1': {'velocity': '0', 'temperature': '302', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T305_U0_B1_M1': {'velocity': '0', 'temperature': '305', 'bowen_ratio': '1', 'microphysic': '1'},
    'RCE_T295_U0_B1_M1': {'velocity': '0', 'temperature': '295', 'bowen_ratio': '1', 'microphysic': '1'}

}


def generate_simulation_paths(velocity: str, temperature: str, bowen_ratio: str, microphysic: str) -> dict:
    """
    Generate the paths to the 3D, 2D, and 1D datasets based on simulation parameters.
    
    Parameters:
    - velocity (str): Velocity value, e.g., '0'.
    - temperature (str): Temperature value, e.g., '300'.
    - bowen_ratio (str): Bowen ratio value, e.g., '1'.
    - microphysic (str): Microphysics option, e.g., '1'.
    
    Returns:
    - dict: Dictionary containing paths for 'path_3d', 'path_2d', and 'path_1d'.
    """
    # Define base directory (could be parameterized if needed)
    base_dir = '/burg/glab_new/users/sga2133/SAM_simulation_storage'
    
    # Folder name following the pattern with velocity, temperature, bowen_ratio, and microphysic
    folder_name = f'RCE_T{temperature}_U{velocity}_B{bowen_ratio}_M{microphysic}'
    
    # Base simulation path
    base_simulation_path = os.path.join(base_dir, folder_name, 'WORK/NETCDF_files')
    
    # Construct the full paths to 3D, 2D, and 1D datasets
    path_3d = os.path.join(base_simulation_path, '3D/dataset_3d.nc')
    path_2d = os.path.join(base_simulation_path, f'2D/RCE_T{temperature}_U{velocity}_SAM{microphysic}MOM_B{bowen_ratio}_128x128x64_64.2Dcom_1.nc')
    path_1d = os.path.join(base_simulation_path, f'1D/RCE_T{temperature}_U{velocity}_SAM{microphysic}MOM_B{bowen_ratio}_128x128x64.nc')
    
    # Return paths in a dictionary
    return {
        'path_3d': path_3d,
        'path_2d': path_2d,
        'path_1d': path_1d
    }


list_files = ['RCE_T300_U0_B0.1_M1',
    'RCE_T300_U0_B0.5_M1',
    'RCE_T300_U0_B1_M1',
    'RCE_T300_U10_B1_M1',
    'RCE_T300_U20_B1_M1',
    'RCE_T300_U2.5_B1_M1',
    'RCE_T300_U5_B1_M1',
    'RCE_T302_U0_B1_M1',
    'RCE_T305_U0_B1_M1',
    'RCE_T295_U0_B1_M1']




# Main execution block
if __name__ == "__main__":
    for file in tqdm(list_files):
        print('Run', file)
        parameters = data_dict[file]
        paths = generate_simulation_paths(**parameters)
        path_netcdf = paths['path_3d'] #3d
        path_netcdf_1d = paths['path_1d'] #3d
        
        storage_path = f'/burg/glab_new/users/sga2133/image_storage_video/mass_flux_{file}'
        os.makedirs(storage_path, exist_ok=True)

        ds_1d=xr.open_dataset(path_netcdf_1d)
        ds = xr.open_dataset(path_netcdf)
        xr_var_2d = ds.TABS[:,0,:,:]

        xr_var_3d = pySAMetrics.utils.expand_array_to_tzyx_array(input_array = ds_1d.RHO[-481:].values,
                                                     final_shape = (481,64,128,128),
                                                     time_dependence=True)*ds.W.values

        #xr_var_3d = ds.QN[ :, :, :] + ds.QP[ :, :, :]

        x= ds.x.values
        y= ds.y.values
        z= ds.z.values




        outname = os.path.join(storage_path,f'img_{file}')
        get_sequence_images(i_start=5, 
                            i_stop=300, 
                            xr_var_2d=xr_var_2d, 
                            xr_var_3d=xr_var_3d,
                            x=x, 
                            y=y, 
                            z=z,  
                            outname=outname)
        


