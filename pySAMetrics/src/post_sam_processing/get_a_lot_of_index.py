import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm

from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths
from pySAMetrics.basic_variables import set_basic_variables_from_dataset
from pySAMetrics.coldpool_tracking import get_coldpool_tracking_images
from pySAMetrics.diagnostic_fmse_v2 import get_isentropic_dataset
from script_simu_high_Res_long import data_dict, load_simulation
import multiprocessing as mp

# Variables to extract
variables = ['Prec', 'W500', 'LHF', 'IWP', 'LWNT']

# Initialize storage
all_vars_mean = {var: [] for var in variables}
all_vars_var = {var: [] for var in variables}

output_dir = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/'
list_files = [f'split_{i}' for i in range(4, 40)]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for i_file, file in tqdm(enumerate(list_files, start=4)):
        print('Run', file)
        parameters = data_dict[file]
        simu = load_simulation(parameters, i=i_file)

        print(simu.name)

        for var in variables:
            print(f"Extracting {var}...")
            try:
                data = getattr(simu.dataset_2d, var).values  # shape: (nt, nx, ny)
                spatial_mean = np.mean(data, axis=(-1, -2))  # shape: (nt,)
                spatial_var = np.var(data, axis=(-1, -2))    # shape: (nt,)
                all_vars_mean[var].append(spatial_mean)
                all_vars_var[var].append(spatial_var)
            except Exception as e:
                print(f"Failed to extract {var} for {file}: {e}")
                continue

    # Save all data
    for var in variables:
        if all_vars_mean[var]:
            mean_data = np.concatenate(all_vars_mean[var], axis=0)
            var_data = np.concatenate(all_vars_var[var], axis=0)

            mean_path = os.path.join(output_dir, f'mean_{var}.npy')
            var_path = os.path.join(output_dir, f'var_{var}.npy')

            print(f"Saving mean of {var} to {mean_path}")
            np.save(mean_path, mean_data)

            print(f"Saving variance of {var} to {var_path}")
            np.save(var_path, var_data)
        else:
            print(f"No data collected for {var}. Nothing to save.")
