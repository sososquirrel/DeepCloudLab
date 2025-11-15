import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle

# Plotly (for future use)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimage import measure

# pySAMetrics modules
import pySAMetrics
from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths
from pySAMetrics.ColdPool import ColdPool, extract_cold_pools

# Define data dict
data_dict = {}
for i in range(1, 49):
    data_dict[f'split_{i}'] = {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': str(i)
    }

# Function to load a simulation
def load_simulation(simu_parameters, i=1000, path_raw_data='/Volumes/LaCie/000_POSTDOC_2025/long_high_res'):
    try:
        if i != 1000:
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

# List of files to process
list_files = [f'split_{i}' for i in range(4, 40)]

mp.set_start_method("spawn", force=True)

# ===== NEW: flat list of CP dict entries =====
all_cold_pools = []

# global timestep cursor
global_t_cursor = 0

# Main loop
for i_file, file in tqdm(enumerate(list_files, start=4)):
    print('Run', file)
    parameters = data_dict[file]

    simu = load_simulation(parameters, i=i_file)
    if simu is None:
        print(f"Skipping {file} due to load failure.")
        continue

    print('Simulation name:', simu.name)

    simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')

    # Extract cold pools
    label_array = simu.dataset_computed_2d.CP_LABELS.values
    qv_array = simu.dataset_3d.QV[:, 0].values
    cold_pools = extract_cold_pools(label_array, qv_array)

        # For each cold pool object in this simulation
    for cp in cold_pools:
        # cp.cluster['timesteps'] is a list of local timesteps
        for local_ts, size in zip(cp.cluster['timesteps'], cp.cluster['sizes']):
            entry = {
                "ts": global_t_cursor + local_ts,
                "sim": simu.name,
                "area": size,
                "anomaly_qv": cp.anomaly_qv,
                "max_anomaly_qv": cp.max_anomaly_qv
            }
            all_cold_pools.append(entry)

    # Advance cursor after finishing this simulation
    global_t_cursor += label_array.shape[0]

# Save output
with open("all_cold_pools.pkl", "wb") as f:
    pickle.dump(all_cold_pools, f)

print("✅ Saved all_cold_pools.pkl with global time mapping!")
print(f"Total timesteps = {global_t_cursor}")
print(f"Total cold pools = {len(all_cold_pools)}")
