#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate the minimal number of clusters (states) needed so that
the dataset naturally satisfies the Dice constraints:

   same–cluster:   dice_distance <  0.25
   diff–cluster:   dice_distance >  0.40

But instead of requiring 100% satisfaction, we allow a tolerance:
e.g. 95% of constraints must be satisfied.

Also plots satisfaction % vs. K.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# IMPORT YOUR UTILITY FUNCTIONS
# ---------------------------------------------------------------------
from model import (
    inv_log_signed,
    create_image_from_flat_tensor_torch,
    make_three_masks_torch,
    vectorized_macro_dice_from_masks
)

# ---------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------
DATA_PATH = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dice thresholds
DICE_THR_COH = 0.25
DICE_THR_SEP = 0.40

# Max number of clusters to test
MAX_CLUSTERS = 20

# Random validation pairs
NUM_PAIRS = 5000

# Validation threshold (e.g. 0.95 → 95% of constraints satisfied)
VALIDATION_THRESHOLD = 0.95

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
print("Loading dataset...")
data_np = np.load(DATA_PATH)
data_torch = torch.tensor(data_np, dtype=torch.float32).to(DEVICE)

N, D = data_torch.shape
print(f"Dataset shape: N={N}, D={D}")

# ---------------------------------------------------------------------
# TRANSFORM INTO IMAGES + 3 MASKS
# ---------------------------------------------------------------------
print("Creating images and masks...")
with torch.no_grad():
    imgs = create_image_from_flat_tensor_torch(inv_log_signed(data_torch))
    masks = make_three_masks_torch(imgs)       # shape [N, 3, 48, 48]
    masks = masks.float()

# ---------------------------------------------------------------------
# DICE DISTANCE FUNCTION
# ---------------------------------------------------------------------
def compute_dice_distances(idx_i, idx_j):
    """
    Returns dice distance = 1 - dice coefficient
    for each pair (idx_i[t], idx_j[t]).
    """
    mi = masks[idx_i]   # [num_pairs,3,48,48]
    mj = masks[idx_j]
    dice_vals = vectorized_macro_dice_from_masks(mi, mj)
    return 1.0 - dice_vals


# ---------------------------------------------------------------------
# CHECK CLUSTER VALIDITY (with % satisfied)
# ---------------------------------------------------------------------
def compute_constraint_satisfaction(labels):
    """
    Returns:
        percent_satisfied (float in [0,1])
        percent_coh (same-cluster constraints satisfied)
        percent_sep (diff-cluster constraints satisfied)
    """

    N = len(labels)
    idx_i = np.random.randint(0, N, size=NUM_PAIRS)
    idx_j = np.random.randint(0, N, size=NUM_PAIRS)

    same = (labels[idx_i] == labels[idx_j])

    # Compute distances in batches handled by masks
    dice_dist = compute_dice_distances(
        torch.tensor(idx_i, device=DEVICE),
        torch.tensor(idx_j, device=DEVICE)
    ).cpu().numpy()

    # SAME-CLUSTER constraints (coherence)
    coh_mask = same
    if coh_mask.any():
        coh_dist = dice_dist[coh_mask]
        coh_ok = (coh_dist < DICE_THR_COH).mean()
    else:
        coh_ok = 1.0  # no same-cluster samples → trivially satisfied

    # DIFF-CLUSTER constraints (separation)
    sep_mask = ~same
    if sep_mask.any():
        sep_dist = dice_dist[sep_mask]
        sep_ok = (sep_dist > DICE_THR_SEP).mean()
    else:
        sep_ok = 1.0

    # Combine
    n_coh = coh_mask.sum()
    n_sep = sep_mask.sum()
    percent_satisfied = (coh_ok*n_coh + sep_ok*n_sep) / (n_coh + n_sep)

    return percent_satisfied, coh_ok, sep_ok


# ---------------------------------------------------------------------
# MAIN LOOP: Sweep K
# ---------------------------------------------------------------------
print("\nEstimating required number of clusters...\n")

flat_np = data_np  # KMeans on original vectors

K_values = []
sat_values = []     # total constraint satisfaction
coh_values = []     # coherence satisfaction
sep_values = []     # separation satisfaction

best_k = None

for k in range(1, MAX_CLUSTERS + 1):

    print(f" → Testing K = {k} clusters...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(flat_np)

    pct, coh_pct, sep_pct = compute_constraint_satisfaction(labels)

    K_values.append(k)
    sat_values.append(pct)
    coh_values.append(coh_pct)
    sep_values.append(sep_pct)

    print(
        f"   Total satisfaction: {pct*100:5.2f}% "
        f"(same={coh_pct*100:5.2f}%, diff={sep_pct*100:5.2f}%)"
    )

    if pct >= VALIDATION_THRESHOLD:
        print(f"\n✔ Found minimal K satisfying threshold: K = {k}\n")
        best_k = k
        break
    else:
        print(f"   K={k} does NOT satisfy threshold ({VALIDATION_THRESHOLD*100:.1f}%).")

if best_k is None:
    print("\n✘ No K ≤ MAX_CLUSTERS satisfies the constraints threshold.")
else:
    print(f"★ Minimal number of states required: {best_k}")


# ---------------------------------------------------------------------
# FINAL PLOT
# ---------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot(K_values, sat_values, marker='o', label="Total satisfaction")
plt.axhline(VALIDATION_THRESHOLD, linestyle="--", linewidth=2, label="Threshold")

plt.xlabel("Number of clusters K")
plt.ylabel("Constraint satisfaction (%)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.title("Constraint Satisfaction vs. K")

plt.tight_layout()
plt.show()
