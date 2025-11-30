import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from model import VAE_HMM, compute_hmm_vae_loss
from model import dice_distance_loss_random_pairs_from_true_x, vectorized_macro_dice_from_masks
from model import inv_log_signed, create_image_from_flat_tensor_torch, make_three_masks_torch


import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# ------------------------------------------------------
# Straight-through one-hot (Unchanged)
def straight_through_one_hot_from_probs(probs):
    idx = probs.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
    one_hot_st = (one_hot - probs).detach() + probs
    return one_hot_st, idx.squeeze(1)

# ------------------------------------------------------
# GLOBALS and Helpers
DATA_FOLDER_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
valid_indices = np.load(os.path.join(DATA_FOLDER_PATH, "valid_indices.npy"))
valid_indices_torch = torch.from_numpy(valid_indices).long()

IMAGE_SIZE = 48
DATA_RANGE = 10.0

# ============================================================
# Hyperparameters (same as your last snippet
LATENT_DIM = 8
HIDDEN_DIM = 512
NUM_STATES = 10
BETA_KL = 0.008
GAMMA_HMM = 0.3
GAMMA_ENTROPY = 0
NUM_PAIRS_TRIPLET = 256

LAMBDA_DICE = 200.0   # keep if you want dice reg
LR = 5e-5
BATCH_SIZE = 256
NUM_EPOCHS = 256
CLIP_GRAD = 1.0

DICE_THR_COH = 0.25
DICE_THR_SEP = 0.35


OUT_DIR = '/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/src/VAE-HMM-DiceSep-FB/runs/exp5'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Dataset (unchanged)
DATA_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy'
data = np.load(DATA_PATH)
full_tensor = torch.tensor(data, dtype=torch.float32)
input_dim = data.shape[1]

indices = np.arange(len(data))
np.random.shuffle(indices)
train_idx = indices[:int(0.90 * len(indices))]
val_idx   = indices[int(0.90 * len(indices)):int(0.999 * len(indices))]

class NextStepDataset(Dataset):
    def __init__(self, data, idxs):
        self.data = data
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        if idx + 1 >= len(self.data):
            idx = len(self.data) - 2
        return self.data[idx], self.data[idx + 1]

train_loader = DataLoader(NextStepDataset(full_tensor, train_idx), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(NextStepDataset(full_tensor, val_idx), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

model = VAE_HMM(input_dim, HIDDEN_DIM, LATENT_DIM, NUM_STATES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Loss tracker (improved logging: accumulate sums and counts to avoid last-batch bias)
class EpochAccumulator:
    def __init__(self):
        # each key stores {"sum": float, "count": int} for scalar metrics
        self.scalars = {}
        self.other = {}
        self.num_samples = 0
    def add_scalar(self, name, value, n=1):
        if name not in self.scalars:
            self.scalars[name] = {"sum": 0.0, "count": 0.0}
        self.scalars[name]["sum"] += float(value) * n
        self.scalars[name]["count"] += n
    def set_other(self, name, value):
        self.other[name] = value
    def get_epoch_metrics(self):
        out = {}
        for k, v in self.scalars.items():
            out[k] = v["sum"] / max(1e-12, v["count"])
        out.update(self.other)
        return out

loss_tracker = {"train": [], "val": []}

def display_epoch_metrics(split, epoch_metrics):
    keys = ["total", "recon", "kl_scaled", "hmm_scaled", "dice_loss_scaled", "entropy_unscaled"]
    msg = f"[{split}] "
    for k in keys:
        if k in epoch_metrics:
            msg += f"{k}:{epoch_metrics[k]:.4f} | "
    if "trans_acc" in epoch_metrics:
        msg += f"TRANS_ACC:{epoch_metrics['trans_acc']:.4f} | "
    '''if "triplet_ratio" in epoch_metrics:
        msg += f"TRIPLET:{epoch_metrics['triplet_ratio']:.4f} | "'''
    print(msg)

# ============================================================
# Training Loop (main) - speed-optimized and with correct dice on TRUE x
best_val = float('inf')
best_ckpt_path = os.path.join(OUT_DIR, "best_hmm_vae_checkpoint.pt")

# ============================================================
# Training Loop (CORRECTED)
# ============================================================
for epoch in range(NUM_EPOCHS):
    model.train()
    train_acc = EpochAccumulator()
    total_samples_train = 0

    for x_t, x_tp1 in train_loader:
        batch_size = x_t.shape[0]
        total_samples_train += batch_size

        x_t = x_t.to(DEVICE, non_blocking=True)
        x_tp1 = x_tp1.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        out_t = model(x_t)
        out_tp1 = model(x_tp1)

        loss_dict = compute_hmm_vae_loss(
            x_t, x_tp1, out_t, out_tp1,
            beta_kl=BETA_KL, gamma_hmm=GAMMA_HMM,
            lambda_entropy=GAMMA_ENTROPY
        )

        dice_pen = dice_distance_loss_random_pairs_from_true_x(
            x_flat=out_t["input_x"],
            s_probs=out_t["s_probs"],
            num_pairs=NUM_PAIRS_TRIPLET,
            thr_coh=DICE_THR_COH,
            thr_sep=DICE_THR_SEP,
            device=DEVICE
        )

        loss_dict["dice_loss_unscaled"] = dice_pen.detach()
        loss_dict["dice_loss_scaled"] = LAMBDA_DICE * dice_pen
        loss_dict["total"] = loss_dict["total"] + loss_dict["dice_loss_scaled"]

        with torch.no_grad():
            # 1. Transition Accuracy
            s_t = out_t["s_argmax"]
            # Predict next state based on transition matrix
            s_tp1_pred = out_t["trans_mat"][s_t.long()].argmax(dim=1)
            trans_accuracy = (s_tp1_pred == out_tp1["s_argmax"]).float().mean()

            # 2. Triplet Ratio
            B = batch_size
            if B < 2:
                triplet_ratio = torch.tensor(0.0, device=DEVICE)
            else:

                idx_i = torch.randint(0, B, (NUM_PAIRS_TRIPLET,), device=DEVICE)
                idx_j = torch.randint(0, B, (NUM_PAIRS_TRIPLET,), device=DEVICE)

                imgs = create_image_from_flat_tensor_torch(inv_log_signed(out_t["input_x"])).to(DEVICE)
                masks = make_three_masks_torch(imgs)

                masks_i = masks[idx_i]
                masks_j = masks[idx_j]
                dice_vals = vectorized_macro_dice_from_masks(masks_i, masks_j)
                dice_distances = 1.0 - dice_vals

                same_state_mask = (out_t["s_argmax"][idx_i] == out_t["s_argmax"][idx_j])

                respect_A = (dice_distances > DICE_THR_SEP) & (~same_state_mask)
                respect_B = (dice_distances < DICE_THR_COH) & (same_state_mask)
                applies = (dice_distances > DICE_THR_SEP) | (dice_distances < DICE_THR_COH)
                respects = torch.where(~applies, torch.ones_like(applies, dtype=torch.bool), (respect_A | respect_B))
                triplet_ratio = respects.float().mean()

            # 3. State Counts
            num_states = out_t["trans_mat"].shape[0]
            all_states = torch.cat([out_t["s_argmax"], out_tp1["s_argmax"]]).long()
            state_counts = torch.bincount(all_states, minlength=num_states)

            # --- LOGGING CORRECTED ---
            # Use add_scalar for metrics you want averaged
            train_acc.add_scalar("total", loss_dict["total"].detach().item(), n=batch_size)
            train_acc.add_scalar("recon", loss_dict["recon"].detach().item(), n=batch_size)
            train_acc.add_scalar("kl_scaled", loss_dict["kl_scaled"].detach().item(), n=batch_size)
            train_acc.add_scalar("hmm_scaled", loss_dict["hmm_scaled"].detach().item(), n=batch_size)
            train_acc.add_scalar("dice_loss_scaled", (LAMBDA_DICE * dice_pen).detach().item(), n=batch_size)
            train_acc.add_scalar("entropy_unscaled", loss_dict["entropy_unscaled"].detach().item(), n=batch_size)
            
            # FIXED: Use add_scalar here with simple names
            train_acc.add_scalar("trans_acc", trans_accuracy.item(), n=batch_size)
            train_acc.add_scalar("triplet_ratio", triplet_ratio.item(), n=batch_size)
            
            # Keep set_other for state counts (since it's an array, not a scalar)
            train_acc.set_other("state_counts", state_counts.cpu().numpy())

        loss_dict["total"].backward()
        clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

    # --- END OF TRAIN EPOCH CORRECTED ---
    # get_epoch_metrics auto-averages everything added via add_scalar
    epoch_train_metrics = train_acc.get_epoch_metrics() 
    
    # We only need to manually attach state_counts (because it's in 'other')
    # Note: trans_acc and triplet_ratio are ALREADY in epoch_train_metrics now!
    epoch_train_metrics["state_counts"] = train_acc.other.get("state_counts", np.zeros(NUM_STATES, dtype=np.int64)).tolist()

    loss_tracker["train"].append(epoch_train_metrics)
    display_epoch_metrics("train", epoch_train_metrics)

    # ============================================================
    # Validation Loop (CORRECTED)
    # ============================================================
    model.eval()
    val_acc = EpochAccumulator()
    total_samples_val = 0
    with torch.no_grad():
        for x_t, x_tp1 in val_loader:
            batch_size = x_t.shape[0]
            total_samples_val += batch_size
            x_t = x_t.to(DEVICE, non_blocking=True)
            x_tp1 = x_tp1.to(DEVICE, non_blocking=True)

            out_t = model(x_t)
            out_tp1 = model(x_tp1)

            loss_dict = compute_hmm_vae_loss(
                x_t, x_tp1, out_t, out_tp1,
                beta_kl=BETA_KL, gamma_hmm=GAMMA_HMM,
                lambda_entropy=GAMMA_ENTROPY
            )

            dice_pen = dice_distance_loss_random_pairs_from_true_x(
                x_flat=out_t["input_x"],
                s_probs=out_t["s_probs"],
                num_pairs=NUM_PAIRS_TRIPLET,
                thr_coh=DICE_THR_COH,
                thr_sep=DICE_THR_SEP,
                device=DEVICE
            )

            loss_dict["dice_loss_unscaled"] = dice_pen
            loss_dict["dice_loss_scaled"]   = LAMBDA_DICE * dice_pen
            loss_dict["total"] = loss_dict["total"] + loss_dict["dice_loss_scaled"]

            # metrics
            s_t = out_t["s_argmax"]
            s_tp1_pred = out_t["trans_mat"][s_t.long()].argmax(dim=1)
            trans_accuracy = (s_tp1_pred == out_tp1["s_argmax"]).float().mean()

            # triplet ratio
            B = batch_size
            if B < 2:
                triplet_ratio = torch.tensor(0.0, device=DEVICE)
            else:
                idx_i = torch.randint(0, B, (NUM_PAIRS_TRIPLET,), device=DEVICE)
                idx_j = torch.randint(0, B, (NUM_PAIRS_TRIPLET,), device=DEVICE)
                imgs = create_image_from_flat_tensor_torch(inv_log_signed(out_t["input_x"])).to(DEVICE)
                masks = make_three_masks_torch(imgs)
                masks_i = masks[idx_i]
                masks_j = masks[idx_j]
                dice_vals = vectorized_macro_dice_from_masks(masks_i, masks_j)
                dice_distances = 1.0 - dice_vals
                same_state_mask = (out_t["s_argmax"][idx_i] == out_t["s_argmax"][idx_j])
                respect_A = (dice_distances > DICE_THR_SEP) & (~same_state_mask)
                respect_B = (dice_distances < DICE_THR_COH) & (same_state_mask)
                applies = (dice_distances > DICE_THR_SEP) | (dice_distances < DICE_THR_COH)
                respects = torch.where(~applies, torch.ones_like(applies, dtype=torch.bool), (respect_A | respect_B))
                triplet_ratio = respects.float().mean()

            num_states = out_t["trans_mat"].shape[0]
            all_states = torch.cat([out_t["s_argmax"], out_tp1["s_argmax"]]).long()
            state_counts = torch.bincount(all_states, minlength=num_states)

            # --- LOGGING CORRECTED (VAL) ---
            val_acc.add_scalar("total", loss_dict["total"].detach().item(), n=batch_size)
            val_acc.add_scalar("recon", loss_dict["recon"].detach().item(), n=batch_size)
            val_acc.add_scalar("kl_scaled", loss_dict["kl_scaled"].detach().item(), n=batch_size)
            val_acc.add_scalar("hmm_scaled", loss_dict["hmm_scaled"].detach().item(), n=batch_size)
            val_acc.add_scalar("dice_loss_scaled", (LAMBDA_DICE * dice_pen).detach().item(), n=batch_size)
            val_acc.add_scalar("entropy_unscaled", loss_dict["entropy_unscaled"].detach().item(), n=batch_size)
            
            # FIXED: Use add_scalar so we get the epoch average, not just the last batch
            val_acc.add_scalar("trans_acc", trans_accuracy.item(), n=batch_size)
            val_acc.add_scalar("triplet_ratio", triplet_ratio.item(), n=batch_size)
            val_acc.set_other("state_counts", state_counts.cpu().numpy())

    epoch_val_metrics = val_acc.get_epoch_metrics()
    # Again, trans_acc and triplet_ratio are already in epoch_val_metrics via add_scalar
    epoch_val_metrics["state_counts"] = val_acc.other.get("state_counts", np.zeros(NUM_STATES, dtype=np.int64)).tolist()

    loss_tracker["val"].append(epoch_val_metrics)
    display_epoch_metrics("val", epoch_val_metrics)

    # (Checkpoints logic remains same...)
    last_val_total = epoch_val_metrics.get("total", float('inf'))
    if last_val_total < best_val:
        best_val = last_val_total
        torch.save({"state_dict": model.state_dict()}, best_ckpt_path)
        print(f"Saved best (val={best_val:.6f})")


print("Training done.")