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



def load_simulation(simu_parameters):
    """
    Load and run the simulation for the given parameters.

    Parameters:
    - simu_parameters (dict): Dictionary containing simulation parameters.

    Returns:
    - Simulation object or None if failed to load.
    """
    try:
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
"""
list_files = ['RCE_T300_U10_B1_M1',
            'RCE_T300_U5_B1_M1',
            'RCE_T300_U2.5_B1_M1',
            'RCE_T300_U20_B1_M1',]


# Main execution block
if __name__ == "__main__":
    for file in tqdm(list_files):
        print('Run', file)
        parameters = data_dict[file]
        simu = load_simulation(parameters)
        
        if simu:
            print('# Basic Variables')
            simu.set_basic_variables_from_dataset()

            print('# Cold Pool Tracking')
            variable_images = simu.dataset_3d.TABS[:, 0].values
            q_inf, q_sup = np.quantile(variable_images, [0.05, 0.1])
            simu.get_coldpool_tracking_images(variable_images=variable_images, low_threshold=q_sup, high_threshold=q_inf)

            print('# CAPE Calculation')
            simu.get_cape()

            print('# Isentropic Coordinates')
            simu.get_isentropic_dataset()

            print('# Save Everything')
            simu.save(backup_folder_path='/burg/glab_new/users/sga2133/pySAMetrics_saved_simulations')



