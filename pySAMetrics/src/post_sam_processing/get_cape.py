import os
import numpy as np
from tqdm import tqdm

from script_simu_high_Res_long import data_dict, load_simulation
from pySAMetrics.cape_analysis import get_cape  # <--- adjust filename if needed

# Where saved simulations live
save_dir = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu"

# Which splits to process
list_files = [f"split_{i}" for i in range(4, 40)]
# list_files = ["split_4"]  # debugging

if __name__ == "__main__":

    for i_file, file in tqdm(enumerate(list_files, start=4)):
        print(f"\n▶️ Run {file}")

        # Load parameters and simulation
        parameters = data_dict[file]
        simu = load_simulation(parameters, i=i_file)

        # Load previous computed datasets (if any)
        simu.load(backup_folder_path=save_dir)

        # Create `dataset_computed_2d` if missing
        if not hasattr(simu, "dataset_computed_2d") or simu.dataset_computed_2d is None:
            print("⚠️ Creating empty dataset_computed_2d container")
            simu.dataset_computed_2d = simu.dataset_2d.copy(deep=False).isel(time=slice(0,0))

        # Skip if CAPE already computed
        if "CAPE" in simu.dataset_computed_2d:
            print("✅ CAPE already present — skipping.")
            continue

        # Compute CAPE and attach inside simulation
        get_cape(
            simu,
            temperature="TABS",
            humidity_ground="QV",
            pressure="p",
            vertical_array="z",
            parallelize=True,
            set_parcel_ascent_composite_1d=False,
        )

        # Sanity check
        cape = simu.dataset_computed_2d["CAPE"].values
        print(f"✅ CAPE computed — shape {cape.shape}")

        # Save simulation state
        simu.save(save_dir)
        print("💾 Saved into dataset_computed_2d")
