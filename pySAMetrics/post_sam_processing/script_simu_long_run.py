import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm

from pySAMetrics.Simulation import Simulation

# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation import Simulation
from pySAMetrics.utils import generate_simulation_paths

# Initialize the main dictionary
data_dict = {
    'split_1': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_2': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_3': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_4': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_5': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_6': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_7': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_8': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_9': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
    'split_10': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'}

}



def load_simulation(simu_parameters, i=1000):
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
            'path_3d': f'/home/ec2-user/DeepCloudLab/outputs/RCE_splits_v2/3D/split_{i+1}.nc',
            'path_2d': f'/home/ec2-user/DeepCloudLab/outputs/RCE_splits_v2/2D/split_{i+1}.nc',
            'path_1d': f'/home/ec2-user/DeepCloudLab/outputs/RCE_splits_v2/1D/split_{i+1}.nc'
            }

        else:
            paths = generate_simulation_paths(**simu_parameters)

        simu = Simulation(data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
                          **simu_parameters)
        return simu
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the simulation paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
    
    return None

"""
list_files = ['RCE_T300_U0_B0.1_M1','RCE_T300_U0_B0.5_M1',
            'RCE_T300_U0_B1_M1',
            'RCE_T300_U10_B1_M1',
            'RCE_T300_U20_B1_M1',
            'RCE_T300_U2.5_B1_M1',
            'RCE_T300_U5_B1_M1',
            'RCE_T302_U0_B1_M1',
            'RCE_T305_U0_B1_M1',
            'RCE_T295_U0_B1_M1']


list_files = ['RCE_T300_U0_B1_M1',
            'RCE_T300_U10_B1_M1',
            'RCE_T300_U20_B1_M1',
            'RCE_T300_U2.5_B1_M1',
            'RCE_T300_U5_B1_M1',
            'RCE_T302_U0_B1_M1',
            'RCE_T305_U0_B1_M1',
            'RCE_T295_U0_B1_M1']

list_files = ['RCE_T300_U10_B1_M1',
            'RCE_T300_U5_B1_M1',
            'RCE_T300_U2.5_B1_M1',
            'RCE_T300_U20_B1_M1',]
"""

list_files = [f'split_{i}' for i in range(1,11)]

# Main execution block
if __name__ == "__main__":
    for i_file,file in tqdm(enumerate(list_files[1:])):
        print('Run', file)
        parameters = data_dict[file]
        
        simu = load_simulation(parameters, i=i_file)
        print(simu.name)


        if simu:
            simu.load(backup_folder_path=f'/home/ec2-user/DeepCloudLab/processed_classes/long_run_RCE_T300_U0_B1_M1_split_{i_file}')
            #print('# Basic Variables')
            #simu.set_basic_variables_from_dataset()

            print('# Cold Pool Tracking')
            variable_images = simu.dataset_3d.TABS[:, 0].values
            q_inf, q_sup = np.quantile(variable_images, [0.05, 0.1])
            simu.get_coldpool_tracking_images(variable_images=variable_images, low_threshold=q_sup, high_threshold=q_inf)

            #print('# CAPE Calculation')
            #simu.get_cape()

            #print('# Isentropic Coordinates')
            #simu.get_isentropic_dataset()

            print('# Save Everything')
            simu.save(backup_folder_path=f'/home/ec2-user/DeepCloudLab/processed_classes/long_run_RCE_T300_U0_B1_M1_split_{i_file}')


