import numpy as np
from scipy.ndimage import gaussian_filter

# ========== UTILITY FUNCTION ==========
def log_signed(x):
    """Signed logarithmic transform: sign(x) * log(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))

# ========== PARAMETERS ==========
path_input = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/reshaped_rho_w_sum.npy'
path_output = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy'
sigma = 1  # Gaussian smoothing strength

# ========== LOAD AND CENTER DATA ==========
data = np.load(path_input)  # Shape: (T, 48, 48)
data -= data.mean(axis=0)   # Center each pixel over time

# ========== MASK NONZERO REGIONS ==========
T, H, W = data.shape
data_flat = data.reshape(T, -1)  # Shape: (T, H*W)
mask = np.any(data_flat != 0, axis=0)  # Keep pixels with nonzero signal at any time
valid_indices = np.where(mask)[0]
data_masked = data_flat[:, mask]

# ========== GAUSSIAN SMOOTHING ==========
# Reconstruct full field with zeros elsewhere
data_full = np.zeros((T, H * W))
data_full[:, valid_indices] = data_masked
data_full = data_full.reshape(T, H, W)

# Apply Gaussian filter to each frame
data_smoothed = np.array([gaussian_filter(frame, sigma=sigma) for frame in data_full])

# Re-mask the smoothed data
data_smoothed_flat = data_smoothed.reshape(T, -1)[:, valid_indices]

# ========== LOG TRANSFORM AND SAVE ==========
data_log = log_signed(data_smoothed_flat)
np.save(path_output, data_log)
print(f"✅ Saved processed data to: {path_output}")

# ========== SAVE VALID INDICES ==========
valid_indices_path = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/valid_indices.npy'
np.save(valid_indices_path, valid_indices)
print(f"✅ Saved valid indices to: {valid_indices_path}")