#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cmocean
# ---------------------------------------
# Import your model
# ---------------------------------------
from model import VAE_HMM


# ============================================================
#               CONFIGURATION
# ============================================================
DATA_PATH = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
PW_PATH   = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/indexes/var_PW.npy"

MODEL_PATH = '/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp5/best_hmm_vae_checkpoint.pt'
os.makedirs('/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp3/figs/', exist_ok=True)
OUT_PCA    = "/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp2/figs/pca.png"
OUT_PW     = "/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp2/figs/pw_scatter.png"
OUT_TM     = "/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp2/figs/trans_mat.png"

BATCH_SIZE = 256
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
#              LOAD DATASET
# ============================================================
data = np.load(DATA_PATH)
full_tensor = torch.tensor(data, dtype=torch.float32)
input_dim = data.shape[1]

class FullDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data) - 1
    def __getitem__(self, i):
        return self.data[i]

loader = DataLoader(FullDataset(full_tensor), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
#              LOAD MODEL + CHECKPOINT
# ============================================================
print("\n=== Loading model checkpoint ===")

LATENT_DIM = 8
HIDDEN_DIM = 512
NUM_STATES = 10

model = VAE_HMM(input_dim, HIDDEN_DIM, LATENT_DIM, NUM_STATES).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
elif "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)


# ============================================================
#             EXTRACT EMBEDDINGS + STATES + TRANS MAT
# ============================================================
print("\n=== Extracting latent embeddings ===")

all_z = []
all_states = []
all_trans = []

with torch.no_grad():
    for x in loader:
        x = x.to(DEVICE)
        out = model(x)

        all_z.append(out["mu"].cpu().numpy())
        all_states.append(out["s_argmax"].cpu().numpy())
        all_trans.append(out["trans_mat"].cpu().numpy())

embeddings = np.concatenate(all_z, axis=0)
states = np.concatenate(all_states, axis=0)
trans_mat = np.mean(np.stack(all_trans, axis=0), axis=0)  # average transition matrix

print("Embeddings shape:", embeddings.shape)
print("States shape:", states.shape)
print("Transition matrix shape:", trans_mat.shape)


# ============================================================
#                      PCA
# ============================================================
print("\n=== Running PCA ===")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
emb_pca = pca.fit_transform(embeddings)

print("Explained variance ratio:", pca.explained_variance_ratio_)


# ============================================================
#                     PLOT PCA
# ============================================================
print("\n=== Plotting PCA ===")

plt.figure(figsize=(8, 7))
unique, counts = np.unique(states, return_counts=True)
count_dict = {int(u): int(c) for u, c in zip(unique, counts)}

for s in range(NUM_STATES):
    mask = states == s
    n_s = count_dict.get(s, 0)
    plt.scatter(
        emb_pca[mask, 0],
        emb_pca[mask, 1],
        s=6,
        alpha=0.7,
        label=f"State {s} (n={n_s})"
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Latent Embeddings (Colored by HMM State)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PCA, dpi=300)
plt.show()

print(f"PCA saved to {OUT_PCA}")


# ============================================================
#           NEW A — Scatter PW colored by state
# ============================================================
print("\n=== Plotting PCA PW ===")

pw = np.load(PW_PATH)[:len(states)]  # align lengths
print(pw.shape)



plt.figure(figsize=(8, 7))


plt.scatter(
    emb_pca[:,0],
    emb_pca[:,1],
    c=pw,
    s=6,
    cmap='rainbow',
    alpha=0.7,
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Latent Embeddings (Colored by HMM State)")
plt.legend()
plt.tight_layout()
plt.show()

'''
# ============================================================
#           NEW A — Scatter PW colored by state
# ============================================================
print("\n=== Plotting PCA PW ===")



DATA_FOLDER_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
os.path.join(DATA_FOLDER_PATH,'indexes/q99_Prec.npy')
prec = np.load(PW_PATH)[:len(states)]  # align lengths



plt.figure(figsize=(8, 7))


plt.scatter(
    emb_pca[:,0],
    emb_pca[:,1],
    c=prec,
    s=6,
    cmap=cmocean.cm.ice_r,
    alpha=0.7,
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Latent Embeddings (Colored by HMM State)")
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================
#           NEW B — Transition Matrix (imshow)
# ============================================================
print("\n=== Plotting Transition Matrix ===")
#cmap = plt.cm.Blues(10)

plt.figure(figsize=(6, 5))
plt.pcolormesh(trans_mat, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label="P(s_j | s_i)")
plt.xlabel("Next state j")
plt.ylabel("Current state i")
plt.title("Learned Transition Matrix")
plt.tight_layout()
plt.savefig(OUT_TM, dpi=300)
plt.show()

print(f"Transition matrix saved to {OUT_TM}")



# ============================================================
#           NEW: empirical transition-accuracy + Transition Matrix (imshow)
# ============================================================
print("\n=== Computing empirical transition accuracy and plotting Transition Matrix ===")

# compute empirical transition counts from the inferred states sequence
NUM_STATES = NUM_STATES  # already defined above
empirical_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
for t in range(len(states) - 1):
    cur = int(states[t])
    nxt = int(states[t + 1])
    empirical_counts[cur, nxt] += 1

# predicted next-state per current state from learned trans_mat (argmax over j for P(s_j | s_i))
predicted_next = np.argmax(trans_mat, axis=1)  # shape (NUM_STATES,)

# per-state accuracy: for each i, fraction of observed transitions from i that go to predicted_next[i]
row_totals = empirical_counts.sum(axis=1)  # number of times each state appears as current
per_state_acc = np.zeros(NUM_STATES, dtype=float)
for i in range(NUM_STATES):
    if row_totals[i] > 0:
        per_state_acc[i] = empirical_counts[i, predicted_next[i]] / float(row_totals[i])
    else:
        per_state_acc[i] = np.nan  # no observations for this state in the dataset

# overall (weighted) accuracy across all transitions
total_transitions = empirical_counts.sum()
if total_transitions > 0:
    overall_acc = empirical_counts[np.arange(NUM_STATES), predicted_next].sum() / float(total_transitions)
else:
    overall_acc = float("nan")

print("Predicted next states (argmax of learned trans_mat):", predicted_next.tolist())
print("Row totals (observed transitions from each state):", row_totals.tolist())
print("Per-state accuracy (fraction of observed transitions matching predicted next state):")
for i, acc in enumerate(per_state_acc):
    if np.isnan(acc):
        print(f"  State {i}: n_obs=0 -> acc=nan")
    else:
        print(f"  State {i}: n_obs={row_totals[i]:d} -> acc={acc*100:.2f}%")
print(f"Overall transition accuracy (weighted): {overall_acc*100:.2f}% (based on {total_transitions} transitions)")

# Plot transition matrix and annotate per-row accuracy
plt.figure(figsize=(8, 6))
# main matrix - keep same visual as before
plt.pcolormesh(trans_mat, cmap='Blues', vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label("P(s_j | s_i)")

plt.xlabel("Next state j")
plt.ylabel("Current state i")
plt.title(f"Learned Transition Matrix — overall acc {overall_acc*100:.2f}%")

# adjust axes / ticks so the annotation can fit to the right
plt.xlim(0, NUM_STATES + 1.5)  # allow space on right for accuracy text
plt.ylim(0, NUM_STATES)

# annotate each row with accuracy and also show predicted next-state with an arrow marker
for i in range(NUM_STATES):
    y_pos = i + 0.5  # center of the row cell
    # predicted next-state label
    pred_j = int(predicted_next[i])
    # place predicted state index on top of column j (marker)
    # marker position centered in cell (pred_j + 0.5, y_pos), but plotted slightly to the left to not overlap colorbar
    plt.text(pred_j + 0.5, y_pos, "⟶", va='center', ha='center', fontsize=10, alpha=0.9)

    # accuracy text to the right of matrix
    acc_text = "n/a" if np.isnan(per_state_acc[i]) else f"{per_state_acc[i]*100:.1f}%"
    plt.text(NUM_STATES + 0.2, y_pos, acc_text, va='center', ha='left', fontsize=10)

# also add a small legend for the accuracy column
plt.text(NUM_STATES + 0.2, NUM_STATES + 0.2, "Pred acc", va='bottom', ha='left', fontsize=10, fontweight='bold')

plt.tight_layout(rect=(0, 0, 0.92, 1.0))  # leave margin on right for the accuracy labels
plt.savefig(OUT_TM, dpi=300)
plt.show()

print(f"Transition matrix (with per-state accuracies) saved to {OUT_TM}")

'''

# ============================================================
#                 SHOW 10 RANDOM IMAGES PER STATE
# ============================================================

print("\n=== Rendering 10 random samples per HMM state ===")

# ------------------------------------------------------------
# Load helper variables from your model code
# ------------------------------------------------------------
try:
    from model import valid_indices, IMAGE_SIZE
except:
    raise ValueError("Please export valid_indices and IMAGE_SIZE from model.py.")

# ------------------------------------------------------------
# Re-define the image utilities in NumPy (as given)
# ------------------------------------------------------------
def inv_log_signed_np(x: np.ndarray):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def create_image_from_flat_tensor_np(x_flat: np.ndarray):
    """Convert flat NumPy vector back to H×W using valid_indices."""
    if x_flat.ndim == 1:
        x_flat = x_flat[None, :]
    out = np.zeros((x_flat.shape[0], IMAGE_SIZE * IMAGE_SIZE))
    out[:, valid_indices] = x_flat
    return out.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)


# ------------------------------------------------------------
# Create a directory to save cluster images
# ------------------------------------------------------------
# ============================================================
#                 SHOW 10 RANDOM IMAGES PER STATE
# ============================================================

print("\n=== Rendering 10 random samples per HMM state ===")

# ------------------------------------------------------------
# Load helper variables from your model code
# ------------------------------------------------------------
try:
    from model import valid_indices, IMAGE_SIZE
except:
    raise ValueError("Please export valid_indices and IMAGE_SIZE from model.py.")

# ------------------------------------------------------------
# Re-define the image utilities in NumPy (as given)
# ------------------------------------------------------------
def inv_log_signed_np(x: np.ndarray):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def create_image_from_flat_tensor_np(x_flat: np.ndarray):
    """Convert flat NumPy vector back to H×W using valid_indices."""
    if x_flat.ndim == 1:
        x_flat = x_flat[None, :]
    out = np.zeros((x_flat.shape[0], IMAGE_SIZE * IMAGE_SIZE))
    out[:, valid_indices] = x_flat
    return out.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)



# ============================================================
#            SHOW 10 RANDOM IMAGES PER STATE (contourf)
# ============================================================

print("\n=== Rendering 10 contourf images per HMM state ===")

# --- Make sure spatial grid exists ---
try:
    XX, ZZ  # already defined earlier?
except NameError:
    # Fallback grid for IMAGE_SIZE x IMAGE_SIZE
    x = np.arange(IMAGE_SIZE)
    z = np.arange(IMAGE_SIZE)
    XX, ZZ = np.meshgrid(x, z)

# --- Fallback levels (modify as needed) ---
try:
    levels
except NameError:
    levels = np.linspace(-40, 40, 51)

# --- Output directory ---
CLUSTER_CONTOUR_DIR = os.path.join(
    "/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp5/figs",
    "clusters_contour"
)
os.makedirs(CLUSTER_CONTOUR_DIR, exist_ok=True)

N_SHOW = 10

for s in range(NUM_STATES):
    print(f"  - State {s}")

    idx = np.where(states == s)[0]
    if len(idx) == 0:
        print("    (no samples for this state)")
        continue

    chosen = np.random.choice(idx, size=min(N_SHOW, len(idx)), replace=False)

    nrows, ncols = 2, 5
    fig = plt.figure(figsize=(13, 6))

    for k, idx_global in enumerate(chosen):
        ax = plt.subplot(nrows, ncols, k + 1)

        # --- retrieve original flattened data ---
        x_flat_np = full_tensor[idx_global].cpu().numpy()

        # --- invert log-signed ---
        x_inv = inv_log_signed_np(x_flat_np)

        # --- reshape to image ---
        img = create_image_from_flat_tensor_np(x_inv)[0]

        # --- contourf plot ---
        ax.contourf(XX, ZZ, img, cmap='RdBu_r', levels=levels)
        ax.set_title(f"State {s}\nidx={idx_global}", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"State {s} — {len(idx)} samples", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(CLUSTER_CONTOUR_DIR, f"state_{s}_contours.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"    saved → {out_path}")
