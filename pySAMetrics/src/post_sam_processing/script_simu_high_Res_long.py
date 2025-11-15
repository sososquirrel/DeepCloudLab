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
from pySAMetrics.diagnostic_fmse_v2 import get_isentropic_dataset, add_counts_to_isentropic_dataset
import multiprocessing as mp


# Initialize the main dictionary
#data_dict = {
    #'squall_line': {'velocity': '8.0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
     #'control_short': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'}
    #}

data_dict = {}

for i in range(1, 49):
    data_dict[f'split_{i}'] = {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': str(i)
    }



def load_simulation(simu_parameters, i=1000, path_raw_data='/Volumes/LaCie/000_POSTDOC_2025/long_high_res'):
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
#keys = ['squall_line'] #['squall_line', 'control_short']
list_files = [f'split_{i}' for i in range(11,40)]
if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        
        for i_file,file in tqdm(enumerate(list_files, start=11)):
            print('Run', file)
            parameters = data_dict[file]
            
            simu = load_simulation(parameters, i=i_file)
            print(simu.name)
            
            print('hey hey!!!')
            print(simu.name)



            if simu:
                #simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')

                print('# Basic Variables')
                set_basic_variables_from_dataset(simu)

                print('# Cold Pool Tracking')
                try:
                    variable_images = simu.dataset_3d.TABS[:, 0].values
                    q_inf, q_sup = np.quantile(variable_images, [0.05, 0.1])
                    get_coldpool_tracking_images(simulation=simu, variable_images=variable_images, low_threshold=q_sup, high_threshold=q_inf)
                except:
                    print('problem with cp')

                #print('# CAPE Calculation')
                #simu.get_cape()
                
                
                print('# Isentropic Coordinates')
                try:
                    get_isentropic_dataset(simulation=simu)
                except:
                    print('problem with cp')

                print('# Save Everything')
                simu.save(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu', locking_h5=True)

        



