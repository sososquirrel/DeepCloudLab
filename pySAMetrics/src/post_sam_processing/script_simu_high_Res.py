import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm

from pySAMetrics.Simulation_2 import Simulation

# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths

from pySAMetrics.basic_variables import set_basic_variables_from_dataset
from pySAMetrics.coldpool_tracking import get_coldpool_tracking_images
from pySAMetrics.diagnostic_fmse_v2 import get_isentropic_dataset

# Initialize the main dictionary
data_dict = {
    'squall_line': {'velocity': '7.5', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
     'control_short': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'}
    }



def load_simulation(simu_parameters, i=1000, path_raw_data='/Users/sophieabramian/Documents/DeepCloudLab/data/squall_lines'):
    """
    Load and run the simulation for the given parameters.

    Parameters:
    - simu_parameters (dict): Dictionary containing simulation parameters.

    Returns:
    - Simulation object or None if failed to load.
    """
    try:
        if i!=1000:
            paths = {
            'path_3d': os.path.join(path_raw_data, f'3D/split_{i+1}.nc'),
            'path_2d': os.path.join(path_raw_data, f'2D/split_{i+1}.nc'),
            'path_1d': os.path.join(path_raw_data, f'1D/split_{i+1}.nc'),
            }

        else:
            paths = generate_simulation_paths(**simu_parameters, folder_path=path_raw_data)
            print(paths)

        simu = Simulation(data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
                          **simu_parameters)
        return simu
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the simulation paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
    
    return None


# Main execution block
keys = ['control_short'] #['squall_line', 'control_short']
if __name__ == "__main__":
        
        for key in keys:
            parameters = data_dict[key]
            path_data = f'/Volumes/LaCie/000_POSTDOC_2025/{key}'
            simu = load_simulation(parameters, i=1000, path_raw_data=path_data)
            
            print('hey hey!!!')
            print(simu.name)



            if simu:
                #simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/saved_simu')

                print('# Basic Variables')
                set_basic_variables_from_dataset(simu)


                print('# Cold Pool Tracking')
                variable_images = simu.dataset_3d.TABS[:, 0].values
                q_inf, q_sup = np.quantile(variable_images, [0.05, 0.1])
                get_coldpool_tracking_images(simulation=simu, variable_images=variable_images, low_threshold=q_sup, high_threshold=q_inf)


                #print('# CAPE Calculation')
                #simu.get_cape()
                
                
                print('# Isentropic Coordinates')
                get_isentropic_dataset(simulation=simu)

                print('# Save Everything')
                simu.save(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/{key}/saved_simu')

        



