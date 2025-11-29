#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as torch_ssim 
import random 

# =========================================================
# Configuration
# =========================================================
# NOTE: Ensure these paths exist and contain the required files
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
DATA_FOLDER_PATH = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res"
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
IMAGE_SIZE = 48
DATA_RANGE = 20.0 

# Use a fixed seed for reproducible random pairs
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# Load helpers
# ---------------------------------------------------------
# NOTE: These files must exist for the script to run
valid_indices = np.load(os.path.join(DATA_FOLDER_PATH, "valid_indices.npy"))
z_array = np.loadtxt(os.path.join(DATA_FOLDER_PATH, "z_array.txt"))
valid_indices_torch = torch.from_numpy(valid_indices).long().to(DEVICE) # Move index tensor to device

# ---------------------------------------------------------
# Transformation Functions (Separated for NumPy and PyTorch)
# ---------------------------------------------------------

def inv_log_signed_np(x: np.ndarray):
    """Applies the inverse log-signed transformation to a NumPy array."""
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def inv_log_signed_torch(x: torch.Tensor):
    """Applies the inverse log-signed transformation to a PyTorch tensor."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1) 

# ---------------------------------------------------------
# Image Creation Helpers
# ---------------------------------------------------------

def create_image_from_flat_tensor_np(x_flat: np.ndarray):
    """Converts a flat NumPy array back to H x W for plotting."""
    if x_flat.ndim == 1:
        x_flat = x_flat[None, :]
    out = np.zeros((x_flat.shape[0], IMAGE_SIZE * IMAGE_SIZE))
    out[:, valid_indices] = x_flat
    return out.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

def create_image_from_flat_tensor_torch(x_flat: torch.Tensor):
    """
    Converts a batch of flat PyTorch tensors back into a B x 1 x H x W image 
    tensor for SSIM calculation.
    """
    B = int(x_flat.shape[0])
    
    # Initialize output tensor on the same device and dtype as input
    out = torch.zeros(
        B, 
        IMAGE_SIZE * IMAGE_SIZE, 
        dtype=x_flat.dtype, 
        device=x_flat.device
    )
    
    # Scatter the valid indices data into the full image size
    # Uses the valid_indices_torch moved to the correct device
    out[:, valid_indices_torch] = x_flat
    
    # Reshape to B x C x H x W (C=1)
    return out.view(B, 1, IMAGE_SIZE, IMAGE_SIZE)


# =========================================================
# SSIM Distance Function
# =========================================================

def calculate_ssim_distance(imgA: torch.Tensor, imgB: torch.Tensor, data_range: float = DATA_RANGE):
    """
    Computes the SSIM Distance (1.0 - SSIM Score). imgA/imgB must be B x 1 x H x W.
    """
    ssim_score = torch_ssim(
        imgA, imgB, 
        data_range=data_range, 
        size_average=False,  # Return a tensor of shape (B,)
        win_size=7 
    ) 
    
    # SSIM Distance (dissimilarity)
    ssim_distance = 1.0 - ssim_score
    return ssim_distance.item()

# ---------------------------------------------------------
# THREE-LEVEL MASKS
# ---------------------------------------------------------
def make_three_masks(img):
    mask = np.zeros_like(img, dtype=np.int32)
    mask[img < -1.5] = 0
    mask[(img >= -1.5) & (img <= 1.5)] = 1
    mask[img > 1.5] = 2
    return mask

# ---------------------------------------------------------
# Dice metric for each class + macro average
# ---------------------------------------------------------
def dice_for_class(maskA, maskB, class_id):
    A = (maskA == class_id)
    B = (maskB == class_id)
    inter = np.logical_and(A, B).sum()
    A_sum, B_sum = A.sum(), B.sum()
    if A_sum + B_sum == 0:
        return 1.0
    return (2 * inter) / (A_sum + B_sum)

def compute_macro_dice(maskA, maskB):
    dices = [dice_for_class(maskA, maskB, c) for c in [0,1,2]]
    return np.mean(dices)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
data_np = np.load(DATA_PATH)
# Load data as PyTorch tensor and move to device once
data_torch = torch.from_numpy(data_np).float().to(DEVICE) 
print("Dataset:", data_np.shape)

# ---------------------------------------------------------
# Select random pairs
# ---------------------------------------------------------
n_pairs = 6
N = data_np.shape[0]
pairs = list(zip(
    np.random.randint(0, N, n_pairs),
    np.random.randint(0, N, n_pairs)
))

# ---------------------------------------------------------
# Plotting setup
# ---------------------------------------------------------
fig, axes = plt.subplots(2, n_pairs, figsize=(2*n_pairs, 6))

S = np.arange(IMAGE_SIZE)
Z = z_array[:IMAGE_SIZE]
S_plot, Z_plot = np.meshgrid(S, Z)
levels = np.linspace(-10, 10, 40)

# ---------------------------------------------------------
# Loop
# ---------------------------------------------------------
for i, (a, b) in enumerate(pairs):
    # --- 1. Prepare NumPy data for plotting and Dice calculation ---
    
    # Get flat data and apply inverse log using NumPy function
    xA_flat_np = inv_log_signed_np(data_np[a])
    xB_flat_np = inv_log_signed_np(data_np[b])
    
    # Reshape for plotting/Dice
    xA = create_image_from_flat_tensor_np(xA_flat_np)[0]
    xB = create_image_from_flat_tensor_np(xB_flat_np)[0]

    # --- 2. Prepare PyTorch data for SSIM calculation ---
    
    # Get flat data (tensor on device)
    xA_flat_torch = data_torch[a:a+1] 
    xB_flat_torch = data_torch[b:b+1] 
    
    # Apply transformation using the PyTorch function
    xA_transformed_torch = inv_log_signed_torch(xA_flat_torch)
    xB_transformed_torch = inv_log_signed_torch(xB_flat_torch)

    # Create 4D image tensors for SSIM (B x 1 x H x W)
    xA_ssim = create_image_from_flat_tensor_torch(xA_transformed_torch) 
    xB_ssim = create_image_from_flat_tensor_torch(xB_transformed_torch)

    # Calculate SSIM Distance
    ssim_dist = calculate_ssim_distance(xA_ssim, xB_ssim, data_range=DATA_RANGE)
    
    # --- 3. Dice Calculation ---
    mA = make_three_masks(xA)
    mB = make_three_masks(xB)
    dice_macro = compute_macro_dice(mA, mB)
    dice_dist = 1 - dice_macro

    # --- 4. Label and Plotting ---
    label = f"SSIM_d={ssim_dist:.3f}\nDice_d={dice_dist:.3f}"

    axA = axes[0, i]
    axA.contourf(S_plot, Z_plot, xA, levels=levels, cmap="RdBu_r")
    axA.set_title(f"Idx {a}")
    axA.axis("off")

    axB = axes[1, i]
    axB.contourf(S_plot, Z_plot, xB, levels=levels, cmap="RdBu_r")
    axB.set_title(f"Idx {b}\n{label}")
    axB.axis("off")

plt.suptitle("Random pairs – SSIM Distance vs. Dice Distance")
plt.tight_layout()
plt.show()