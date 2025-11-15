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
import multiprocessing as mp

from script_simu_high_Res_long import data_dict, load_simulation


# Create a list to collect reshaped arrays
all_pw= []

output_path = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/var_pw.npy'

list_files = [f'split_{i}' for i in range(4,40)]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


    for i_file, file in tqdm(enumerate(list_files, start=4)):
        print('Run', file)
        parameters = data_dict[file]

        simu = load_simulation(parameters, i=i_file)

        #if simu:
        #    simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')

        print(simu.name)

        print("Extracting PW...")
        try:
            new_variable = np.var(simu.dataset_2d.PW.values, axis=(-1,-2))
            all_pw.append(new_variable)
        except Exception as e:
            print(f"Failed to extract and reshape data for {file}: {e}")
            continue

    # Concatenate and save
    if all_pw:
        print("Concatenating all var pw...")
        final_data = np.concatenate(all_pw, axis=0)
        print(f"Saving to {output_path}")
        np.save(output_path, final_data)
    else:
        print("No data collected. Nothing to save.")
