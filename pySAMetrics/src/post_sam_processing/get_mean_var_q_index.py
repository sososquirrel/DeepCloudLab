import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import multiprocessing as mp

from script_simu_high_Res_long import data_dict, load_simulation

# Variables to extract
variables = ['PW', 'Prec', 'W500', 'LHF', 'IWP', 'LWNT', 'CR']

# Output directory
output_dir = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/indexes'

# Which split files to process
list_files = [f'split_{i}' for i in range(4, 40)]
#list_files = [f'split_{i}' for i in range(4, 6)]

# Storage
stats = {
    var: {
        'mean': [],
        'var': [],
        'norm_var': [],
        'q90': [],
        'q95': [],
        'q99': []
    }
    for var in variables
}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for i_file, file in tqdm(enumerate(list_files, start=4)):
        print('Run', file)
        parameters = data_dict[file]
        simu = load_simulation(parameters, i=i_file)

        simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')


        for var in variables:
            try:
                # --- Try dataset_2d first ---
                if hasattr(simu.dataset_2d, var):
                    data = getattr(simu.dataset_2d, var).values
                # --- Then dataset_computed_2d ---
                elif hasattr(simu.dataset_computed_2d, var):
                    data = getattr(simu.dataset_computed_2d, var).values
                else:
                    raise AttributeError(f"{var} not found in dataset_2d or dataset_computed_2d")

                nt = data.shape[0]

                spatial_mean = np.mean(data, axis=(-1, -2))
                spatial_var  = np.var(data, axis=(-1, -2))
                norm_var = spatial_var / (spatial_mean**2 + 1e-12)
                q90 = np.percentile(data, 90, axis=(-1, -2))
                q95 = np.percentile(data, 95, axis=(-1, -2))
                q99 = np.percentile(data, 99, axis=(-1, -2))

                stats[var]['mean'].append(spatial_mean)
                stats[var]['var'].append(spatial_var)
                stats[var]['norm_var'].append(norm_var)
                stats[var]['q90'].append(q90)
                stats[var]['q95'].append(q95)
                stats[var]['q99'].append(q99)

            except Exception as e:
                print(f"Failed {var} in {file}: {e}")

    # ---- Save ----
    for var in variables:
        for key in stats[var].keys():
            arr = np.concatenate(stats[var][key], axis=0)

            path = os.path.join(output_dir, f'{key}_{var}.npy')
            print(f"Saving {key}_{var} to {path}")
            np.save(path, arr)
