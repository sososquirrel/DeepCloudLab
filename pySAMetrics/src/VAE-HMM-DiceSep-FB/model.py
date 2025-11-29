#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
DATA_RANGE = 20.0

def inv_log_signed(x):
    """Applies the inverse log-signed transformation (works with torch tensors)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def create_image_from_flat_tensor_torch(x_flat: torch.Tensor):
    """
    Converts a batch of flat, D-dimensional tensors (VAE input format) 
    back into a 48x48 image tensor. Output: B x 1 x H x W.
    """
    B = x_flat.shape[0]
    device = x_flat.device
    out = torch.zeros(B, IMAGE_SIZE * IMAGE_SIZE, dtype=x_flat.dtype, device=device)
    out[:, valid_indices_torch.to(device)] = x_flat
    return out.view(B, 1, IMAGE_SIZE, IMAGE_SIZE)

# ============================================================
# Dice utilities (torch, vectorized)
def make_three_masks_torch(imgs: torch.Tensor):
    """
    imgs: B x H x W or B x 1 x H x W
    returns masks: B x H x W (int64) with values {0,1,2}
    thresholds identical to your numpy version: < -1.5 ->0, -1.5..1.5->1, >1.5->2
    """
    if imgs.dim() == 4:
        imgs = imgs[:, 0]
    mask = torch.zeros_like(imgs, dtype=torch.int64)
    mask[imgs < -1.5] = 0
    mask[(imgs >= -1.5) & (imgs <= 1.5)] = 1
    mask[imgs > 1.5] = 2
    return mask

def vectorized_macro_dice_from_masks(maskA: torch.Tensor, maskB: torch.Tensor, eps=1e-8):
    """
    maskA, maskB: B x H x W integer masks with classes {0,1,2}
    returns dice_per_example: B tensor representing mean dice over classes (macro dice)
    Vectorized, runs on GPU.
    """
    B = maskA.shape[0]
    dices = []
    for c in (0, 1, 2):
        A = (maskA == c).view(B, -1).float()
        Bm = (maskB == c).view(B, -1).float()
        inter = (A * Bm).sum(dim=1)
        A_sum = A.sum(dim=1)
        B_sum = Bm.sum(dim=1)
        denom = A_sum + B_sum
        # if denom == 0 -> dice = 1.0
        dice_c = torch.where(denom > 0, (2.0 * inter) / (denom + eps), torch.ones_like(denom))
        dices.append(dice_c)
    dices = torch.stack(dices, dim=1)  # B x 3
    return dices.mean(dim=1)  # B

# ============================================================
# SSIM utilities (global-image SSIM, vectorized)
def global_ssim_batch(x: torch.Tensor, y: torch.Tensor, data_range=DATA_RANGE, eps=1e-8):
    """
    Compute global SSIM between two batches x,y (B x 1 x H x W) using the simplified global formula.
    Returns ssim_scores: B tensor in [-1,1] (practically [0,1]).
    Based on standard SSIM formula:
      mu_x, mu_y = mean over pixels
      var_x, var_y = variance over pixels
      cov_xy = mean((x-mu_x)*(y-mu_y))
      ssim = ((2 mu_x mu_y + C1) * (2 cov + C2)) / ((mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2))
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    B = x.shape[0]
    Npix = float(x.shape[-2] * x.shape[-1])

    mu_x = x.view(B, -1).mean(dim=1)
    mu_y = y.view(B, -1).mean(dim=1)
    x0 = x.view(B, -1)
    y0 = y.view(B, -1)
    var_x = x0.var(dim=1, unbiased=False)
    var_y = y0.var(dim=1, unbiased=False)
    cov = ( (x0 - mu_x.unsqueeze(1)) * (y0 - mu_y.unsqueeze(1)) ).mean(dim=1)

    # constants (as fraction of data_range)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2.0 * mu_x * mu_y + C1) * (2.0 * cov + C2)
    den = (mu_x.pow(2) + mu_y.pow(2) + C1) * (var_x + var_y + C2 + eps)
    ssim = num / (den + eps)
    # clamp for numeric safety
    return torch.clamp(ssim, -1.0, 1.0)

# ============================================================
# Contrastive regularizers computed on true x (vectorized, GPU-friendly)
def dice_distance_loss_random_pairs_from_true_x(
    x_flat: torch.Tensor,
    s_probs: torch.Tensor,
    num_pairs: int = 128,
    thr_coh: float = 0.3,
    thr_sep: float = 0.5,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute dice-based contrastive regularizer using TRUE input x (not recon).
    Vectorized: sample pairs and compute dice distances in batches using torch operations only.
    Returns mean penalty over the sampled pairs.
    """
    B = x_flat.shape[0]
    if B < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    # Convert true x into images (B x 1 x H x W) and masks (B x H x W)
    imgs = create_image_from_flat_tensor_torch(inv_log_signed(x_flat))  # B x 1 x H x W
    masks = make_three_masks_torch(imgs)  # B x H x W

    # Sample pairs (with replacement)
    idx_i = torch.randint(0, B, (num_pairs,), device=device)
    idx_j = torch.randint(0, B, (num_pairs,), device=device)

    # gather masks
    masks_i = masks[idx_i]  # num_pairs x H x W
    masks_j = masks[idx_j]
    # compute macro dice for each pair
    dice_vals = vectorized_macro_dice_from_masks(masks_i, masks_j)  # num_pairs
    dice_distances = 1.0 - dice_vals  # num_pairs

    # p_same between the two examples (use s_probs)
    p_sames = (s_probs[idx_i] * s_probs[idx_j]).sum(dim=1)  # num_pairs
    p_diff = 1.0 - p_sames

    # Penalty terms (contrastive)
    term_cohesion_penalty = p_diff * torch.relu(thr_coh - dice_distances)
    term_separation_penalty = p_sames * torch.relu(dice_distances - thr_sep)

    penalty = (term_cohesion_penalty + term_separation_penalty).mean()
    return penalty

def ssim_contrastive_loss_random_pairs_from_true_x(
    x_flat: torch.Tensor,
    s_probs: torch.Tensor,
    num_pairs: int = 128,
    thr_coh: float = 0.15,   # SSIM thresholds are smaller (SSIM in [0,1])
    thr_sep: float = 0.6,
    device: torch.device = torch.device("cpu"),
    data_range: float = DATA_RANGE
) -> torch.Tensor:
    """
    SSIM-based contrastive regularizer computed on TRUE input x (not recon).
    Uses global SSIM on full image (vectorized).
    thr_coh, thr_sep behave like: if ssim > thr_coh -> same state; if ssim < thr_sep -> different
    We use ssim_distance = 1 - ssim for a similar 'distance' notion.
    """
    B = x_flat.shape[0]
    if B < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    imgs = create_image_from_flat_tensor_torch(inv_log_signed(x_flat)).to(device)  # B x 1 x H x W

    idx_i = torch.randint(0, B, (num_pairs,), device=device)
    idx_j = torch.randint(0, B, (num_pairs,), device=device)

    imgs_i = imgs[idx_i]  # num_pairs x 1 x H x W
    imgs_j = imgs[idx_j]

    ssim_vals = global_ssim_batch(imgs_i, imgs_j, data_range=data_range)  # num_pairs
    ssim_distances = 1.0 - ssim_vals  # num_pairs, in [0,2] but typically [0,1]

    p_sames = (s_probs[idx_i] * s_probs[idx_j]).sum(dim=1)
    p_diff = 1.0 - p_sames

    # Convert thr_coh/sep from SSIM space (we used ssim directly) -> use ssim distances thresholding analogue
    # cohesion condition (ssim close to 1 -> distance small): distance < (1 - thr_coh)
    thr_coh_dist = 1.0 - thr_coh
    thr_sep_dist = 1.0 - thr_sep

    term_cohesion_penalty = p_diff * torch.relu(thr_coh_dist - ssim_distances)
    term_separation_penalty = p_sames * torch.relu(ssim_distances - thr_sep_dist)

    penalty = (term_cohesion_penalty + term_separation_penalty).mean()
    return penalty

# ============================================================
# VAE-HMM model & loss (copied and slightly tidied from your snippet)
class VAE_HMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_states=16, dropout_prob=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_states = num_states

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # State predictor (from z)
        self.state_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
        )

        # Decoder (symmetrical)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_dim)
        )

        self.trans_logits = nn.Parameter(torch.randn(num_states, num_states))

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        logits_state = self.state_predictor(z)
        s_probs = F.softmax(logits_state, dim=-1)
        s_onehot_st, s_argmax = straight_through_one_hot_from_probs(s_probs)

        trans_mat = F.softmax(self.trans_logits, dim=-1)
        recon_x = self.decoder(z)

        return {
            "input_x": x,
            "recon_x": recon_x,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "s_probs": s_probs,
            "s_onehot": s_onehot_st,
            "s_argmax": s_argmax,
            "trans_mat": trans_mat,
            "logits_state": logits_state,
        }

# Loss helpers
def recon_loss_fn(recon_x, x):
    return F.mse_loss(recon_x, x, reduction="mean")

def kl_continuous_z(mu, logvar):
    sigma2 = torch.exp(logvar)
    kl_per_dim = sigma2 + mu.pow(2) - 1.0 - logvar
    kl = 0.5 * torch.sum(kl_per_dim, dim=1)
    return kl.mean()

def hmm_transition_loss(s_probs_t, s_probs_tp1, trans_mat):
    log_trans = torch.log(trans_mat + 1e-10)
    expected_logprob = (s_probs_t.unsqueeze(2) * s_probs_tp1.unsqueeze(1) * log_trans).sum(dim=(1,2))
    return -expected_logprob.mean()

def entropy_regularization(s_probs):
    p = s_probs.mean(dim=0)
    entropy = -(p * (p + 1e-12).log()).sum()
    return entropy

def compute_hmm_vae_loss(
        x_t, x_tp1, out_t, out_tp1,
        beta_kl=1.0, gamma_hmm=1.0,
        lambda_entropy=0.01):
    recon_t = out_t["recon_x"];  mu_t = out_t["mu"];     logvar_t = out_t["logvar"]
    recon_tp1 = out_tp1["recon_x"]; mu_tp1 = out_tp1["mu"]; logvar_tp1 = out_tp1["logvar"]
    s_probs_t = out_t["s_probs"]
    s_probs_tp1 = out_tp1["s_probs"]
    trans_mat = out_t["trans_mat"]

    recon_total = 0.5*(recon_loss_fn(recon_t, x_t) + recon_loss_fn(recon_tp1, x_tp1))
    kl_total    = 0.5*(kl_continuous_z(mu_t, logvar_t) + kl_continuous_z(mu_tp1, logvar_tp1))
    hmm_unscaled = hmm_transition_loss(s_probs_t, s_probs_tp1, trans_mat)
    entropy_term = 0.5*entropy_regularization(s_probs_t)

    total = (recon_total
             + beta_kl*kl_total
             + gamma_hmm*hmm_unscaled
             - lambda_entropy*entropy_term)

    return {
        "total": total,
        "recon": recon_total,
        "kl_scaled": beta_kl*kl_total,
        "hmm_scaled": gamma_hmm*hmm_unscaled,
        "entropy": entropy_term,
        "entropy_scaled": lambda_entropy*entropy_term,
        "entropy_unscaled": entropy_term,
        "kl_unscaled": kl_total,
        "hmm_unscaled": hmm_unscaled,
        "trans_mat": trans_mat.detach(),
    }

# ============================================================
# Metrics utilities (updated to use random-pairs SSIM/Dice evaluation, vectorized)
def compute_metrics_epoch_level(
    accumulators,
    num_samples_seen
):
    """
    Produce epoch-level metrics from accumulators to avoid last-batch bias.
    accumulators is a dict collecting sums/counts per metric across batches.
    """
    out = {}
    # metrics we expect: total_sum, total_count etc.
    # For simple scalar losses we kept sum and count; for lists like state_counts we kept sums
    for k, v in accumulators.items():
        if isinstance(v, dict) and "sum" in v and "count" in v:
            out[k] = v["sum"] / max(1e-12, v["count"])
        else:
            out[k] = v
    # attach sample count
    out["num_samples"] = num_samples_seen
    return out