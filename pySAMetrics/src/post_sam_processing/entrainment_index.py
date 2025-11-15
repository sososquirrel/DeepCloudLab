import os
import numpy as np
from tqdm import tqdm

from script_simu_high_Res_long import data_dict, load_simulation
from pySAMetrics.diagnostic_fmse_v2 import calculate_entrainment_detrainment

# --------------------------
# CONFIG
# --------------------------

# Output directory
output_dir = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/indexes'
os.makedirs(output_dir, exist_ok=True)

# Which split files to process
list_files = [f'split_{i}' for i in range(4, 40)]
# list_files = [f'split_{i}' for i in range(4, 6)]  # debug

# Heights (m) at which to sample entrainment/detrainment
target_heights = np.array([200, 1000, 7000, 10000, 13000])


# --------------------------
# STORAGE
# --------------------------
entrainment_all = []
detrainment_all = []
net_all = []  # E - D


# --------------------------
# MAIN LOOP
# --------------------------

if __name__ == "__main__":

    for i_file, file in tqdm(enumerate(list_files, start=4), desc="Simulations"):
        print("\nRun", file)

        # simulation parameters
        parameters = data_dict[file]

        # load simulation
        simu = load_simulation(parameters, i=i_file)

        # load data fields
        simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')

        # vertical coordinate
        z = simu.dataset_3d.z.values  # meters (nz,)

        # find matching vertical indices once
        idx_levels = np.abs(z[:, None] - target_heights[None, :]).argmin(axis=0)

        nt = simu.nt   # number of timesteps in this split

        # temporary lists for this simulation
        sim_E = []
        sim_D = []
        sim_ED = []

        # loop over timesteps for this simulation
        for t in tqdm(range(nt), desc=file, leave=False):
            try:
                res = calculate_entrainment_detrainment(simu, t, epsilon=1.0)
            except Exception as e:
                print(f"Skipping t={t} because of: {e}")
                continue

            # full vertical profiles
            E = res['E']
            D = res['D']
            ED = res['E_minus_D']

            # append sampled levels
            sim_E.append(E[idx_levels])
            sim_D.append(D[idx_levels])
            sim_ED.append(ED[idx_levels])

        # convert this simulation block to arrays
        sim_E = np.array(sim_E)
        sim_D = np.array(sim_D)
        sim_ED = np.array(sim_ED)

        # append to global lists
        entrainment_all.append(sim_E)
        detrainment_all.append(sim_D)
        net_all.append(sim_ED)

    # --------------------------
    # MERGE ACROSS ALL SIMS
    # --------------------------
    entrainment_all = np.vstack(entrainment_all)  # (N_total_t, 5)
    detrainment_all = np.vstack(detrainment_all)
    net_all = np.vstack(net_all)

    # --------------------------
    # SAVE
    # --------------------------
    np.save(os.path.join(output_dir, "entrainment_levels.npy"), entrainment_all)
    np.save(os.path.join(output_dir, "detrainment_levels.npy"), detrainment_all)
    np.save(os.path.join(output_dir, "net_levels.npy"), net_all)
    np.save(os.path.join(output_dir, "entrainment_heights.npy"), target_heights)

    print("\n✅ Saved entrainment/detrainment index time series!")
    print("Entrainment:", entrainment_all.shape)
    print("Detrainment:", detrainment_all.shape)
    print("Net (E-D):", net_all.shape)
    print("Heights:", target_heights)
