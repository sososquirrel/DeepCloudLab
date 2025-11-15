import pickle
import numpy as np
from tqdm import tqdm

# ========= PARAMETERS ============
pkl_path = "all_cold_pools.pkl"
output_dir = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/"
# =================================

# 1. Load cold pool entries (flat list)
with open(pkl_path, "rb") as f:
    all_cold_pools = pickle.load(f)

print("Loaded cold pool entries:", len(all_cold_pools))

# 2. Determine full time axis
max_ts = max(entry["ts"] for entry in all_cold_pools)
CPI_raw = np.zeros(max_ts + 1)

# 3. Accumulate area at each timestep
for entry in tqdm(all_cold_pools, desc="Building CPI"):
    CPI_raw[entry["ts"]] += entry["area"]

# 4. Normalizations
CPI_norm = CPI_raw / (np.mean(CPI_raw) + 1e-12)
CPI_anom = CPI_raw - np.mean(CPI_raw)
CPI_std  = (CPI_raw - np.mean(CPI_raw)) / (np.std(CPI_raw) + 1e-12)

# 5. Save
np.save(f"{output_dir}/CPI_raw.npy", CPI_raw)
np.save(f"{output_dir}/CPI_norm.npy", CPI_norm)
np.save(f"{output_dir}/CPI_anomaly.npy", CPI_anom)
np.save(f"{output_dir}/CPI_std.npy", CPI_std)

print("✅ Saved CPI time series!")
print("Length:", len(CPI_raw))
print("Mean:", np.mean(CPI_raw), "Std:", np.std(CPI_raw))
