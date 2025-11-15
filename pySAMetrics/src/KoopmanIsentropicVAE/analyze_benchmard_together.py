#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_by_dim_losses_and_eigs.py

Creates per-latent-dim comparison plots:

1) LOSSES (recon, kl, koop):
   - One PNG per loss *and* per latent dim.
   - Overlays all models of that dim.
   - If a model's kl/koop is missing or all zeros -> dashed y=0 baseline for that model.

2) EIGENVALUES:
   - One zoomed PNG per latent dim (xlim=(0.9, 1.0), ylim tight around data).
   - One full-plane PNG per latent dim (for context).
   - One color per model (kind_d{dim}).

Log format expected at: {BENCH_DIR}/logs/{kind}_d{dim}_losses.pkl with keys:
  train_hist: list[dict], e.g. {"recon": ..., "kl": ..., "koop": ...}
  val_hist:   optional list[dict]

Eigenvalues:
  - Loads {OUT_DIR}/{kind}_d{dim}_A.npy if present; else computes from TRAIN latents.
"""

import os, re, glob, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# ========= USER CONFIG (edit here or use CLI) =========
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks_fixed"
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUT_DIR    = os.path.join(BENCH_DIR, "analysis_comp")

KINDS = ["ae", "ae_koop", "sae", "sae_koop", "vae", "kvae",
         "betavae", "wae", "betatcvae", "residualvae", "spectralvae"]
DIMS  = [4, 8, 16]
DT    = 1.0  # only used if you later want discrete->continuous rates

# ========= MODEL IMPORTS (adapt to your repo) =========
from model import VAE
from autoencoder_simple import AE, StochasticAE
from additional_models import BetaVAE, WAE, BetaTCVAE, ResidualVAE, SpectralVAE

def device_auto():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")

def get_model_class(kind):
    if kind in ("ae", "ae_koop"):
        return AE
    if kind in ("sae", "sae_koop"):
        return StochasticAE
    if kind == "betavae":
        return BetaVAE
    if kind == "wae":
        return WAE
    if kind == "betatcvae":
        return BetaTCVAE
    if kind == "residualvae":
        return ResidualVAE
    if kind == "spectralvae":
        return SpectralVAE
    return VAE  # "vae", "kvae"

def parse_outputs(out):
    """Normalize forward outputs across AE/SAE/VAE -> (recon, code, mu, logvar)."""
    if not isinstance(out, (tuple, list)):
        return out, out, None, None
    if len(out) == 2:     # (recon, z)
        recon, z = out
        return recon, z, None, None
    if len(out) == 3:     # (recon, mu, logvar)
        recon, mu, logvar = out
        return recon, mu, mu, logvar
    recon, z, mu, logvar = out[:4]
    code = mu if mu is not None else z
    return recon, code, mu, logvar

@torch.no_grad()
def collect_latents(model, x_tensor, device):
    Zs = []
    for (xb,) in DataLoader(TensorDataset(x_tensor), batch_size=256, shuffle=False):
        xb = xb.to(device)
        out = model(xb)
        _, code, mu, _ = parse_outputs(out)
        Zs.append((mu if mu is not None else code).detach().cpu().numpy())
    return np.concatenate(Zs, axis=0)

def fit_linear_A_from_Z(Z, ridge=1e-6, center=True):
    """Regress Z_{t+1} on Z_t using centered latents; return A and eigenvalues."""
    if center:
        Zc = Z - Z.mean(axis=0, keepdims=True)
    else:
        Zc = Z
    X, Y = Zc[:-1], Zc[1:]
    d = Z.shape[1]
    A = np.linalg.pinv(X.T @ X + ridge*np.eye(d)) @ (X.T @ Y)
    w, V = np.linalg.eig(A)
    return A, w, V

def infer_kind_and_dim(filename):
    base = os.path.basename(filename)
    m_dim = re.search(r"_d(\d+)\.pt$", base)
    d = int(m_dim.group(1)) if m_dim else None
    kind = base.split("_d")[0]
    return kind, d

def load_logs(kind, d, logs_dir):
    path = os.path.join(logs_dir, f"{kind}_d{d}_losses.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

# ========= LOSS PLOTS: per latent dim =========
def plot_losses_by_dim(ckpts, logs_dir, out_dir, loss_keys=("recon","kl","koop")):
    ensure_outdir(out_dir)

    # group checkpoints by dim
    dims = sorted(set(d for _,_,d in ckpts))
    for d in dims:
        ckpts_d = [(p,k,dd) for (p,k,dd) in ckpts if dd == d]
        if not ckpts_d:
            continue

        # colors: one per model (kind_d{d}) within this dim
        tags_d = [f"{k}_d{d}" for _,k,_ in ckpts_d]
        cmap = plt.cm.get_cmap("tab20", max(20, len(tags_d)))
        color_map = {tag: cmap(i % cmap.N) for i, tag in enumerate(tags_d)}

        # preload logs + find max length per loss for dashed baselines
        logs_cache = {}
        max_len_by_loss = {lk: 0 for lk in loss_keys}
        for _, kind, _ in ckpts_d:
            logs = load_logs(kind, d, logs_dir)
            logs_cache[(kind, d)] = logs
            if logs and "train_hist" in logs and len(logs["train_hist"]) > 0:
                for lk in loss_keys:
                    if lk in logs["train_hist"][0]:
                        L = len([e[lk] for e in logs["train_hist"] if lk in e])
                        max_len_by_loss[lk] = max(max_len_by_loss[lk], L)

        # build per-loss figures for this dim
        for lk in loss_keys:
            plt.figure(figsize=(10, 6))
            any_series = False

            for _, kind, _ in ckpts_d:
                tag = f"{kind}_d{d}"
                logs = logs_cache.get((kind, d))
                color = color_map[tag]

                if not logs or "train_hist" not in logs or len(logs["train_hist"]) == 0:
                    # Missing logs altogether: show dashed baseline for kl/koop only
                    if lk in ("kl","koop"):
                        length = max(1, max_len_by_loss[lk])
                        x = np.arange(1, length+1)
                        plt.plot(x, np.zeros_like(x), ls="--", lw=1.6, color=color,
                                 label=f"{tag} (not used/no log)")
                    continue

                train_hist = logs["train_hist"]
                y_tr = [e.get(lk, None) for e in train_hist]
                y_tr = [v for v in y_tr if v is not None]

                if len(y_tr) == 0:
                    # Key missing entirely → treat as not used
                    if lk in ("kl","koop"):
                        length = max(1, max_len_by_loss[lk])
                        x = np.arange(1, length+1)
                        plt.plot(x, np.zeros_like(x), ls="--", lw=1.6, color=color,
                                 label=f"{tag} (not used)")
                    continue

                y_tr = np.asarray(y_tr, dtype=float)
                x_tr = np.arange(1, len(y_tr)+1)
                if lk in ("kl","koop") and np.allclose(y_tr, 0.0, atol=1e-12):
                    # Explicitly recorded but all zeros
                    length = max(1, max_len_by_loss[lk])
                    x = np.arange(1, length+1)
                    plt.plot(x, np.zeros_like(x), ls="--", lw=1.6, color=color,
                             label=f"{tag} (not used)")
                else:
                    any_series = True
                    plt.plot(x_tr, y_tr, lw=2.0, color=color, label=f"{tag} (train)")
                    # optional: add val faintly
                    val_hist = logs.get("val_hist", [])
                    if len(val_hist) and (lk in val_hist[0]):
                        y_va = np.asarray([e.get(lk, None) for e in val_hist if e.get(lk, None) is not None], dtype=float)
                        if y_va.size:
                            x_va = np.arange(1, len(y_va)+1)
                            plt.plot(x_va, y_va, lw=1.2, ls=":", color=color, alpha=0.75, label=f"{tag} (val)")

            plt.title(f"Training curves – {lk} (d={d})")
            plt.xlabel("epoch"); plt.ylabel(lk)
            if any_series and lk in ("recon",):
                plt.yscale("log")
            plt.grid(True, alpha=0.25)
            # dedupe legend
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            plt.legend(uniq.values(), uniq.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
            plt.tight_layout(rect=[0,0,0.80,1])
            out_path = os.path.join(out_dir, f"loss_{lk}_d{d}_ALL.png")
            plt.savefig(out_path, dpi=170)
            plt.close()
            print(f"[ok] saved {out_path}")

# ========= EIGENVALUES: per latent dim =========
def compute_or_load_eigs_for_model(kind, d, out_dir, models_dir, data_path, device):
    tag = f"{kind}_d{d}"
    A_path = os.path.join(out_dir, f"{tag}_A.npy")
    if os.path.exists(A_path):
        try:
            A = np.load(A_path)
            w, _ = np.linalg.eig(A)
            return w
        except Exception:
            pass

    # Fallback: compute from train latents
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"DATA_PATH not found: {data_path}")
    data = np.load(data_path)
    N = len(data)
    train_end = int(0.95 * N)
    train_np  = data[:train_end].copy()
    train_tensor = torch.tensor(train_np, dtype=torch.float32)

    ckpt_path = os.path.join(models_dir, f"{tag}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint for {tag}: {ckpt_path}")

    ModelCls = get_model_class(kind)
    if kind == "betavae":
        model = ModelCls(input_dim=data.shape[1], hidden_dim=512, latent_dim=d, beta=4.0).to(device)
    elif kind == "wae":
        model = ModelCls(input_dim=data.shape[1], hidden_dim=512, latent_dim=d, lambda_mmd=10.0).to(device)
    elif kind == "betatcvae":
        model = ModelCls(input_dim=data.shape[1], hidden_dim=512, latent_dim=d, beta=1.0, gamma=1.0).to(device)
    else:
        model = ModelCls(input_dim=data.shape[1], hidden_dim=512, latent_dim=d).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    Z_train = collect_latents(model, train_tensor, device)
    A, w, _ = fit_linear_A_from_Z(Z_train, ridge=1e-6, center=True)
    # cache A for later runs
    try:
        np.save(A_path, A)
    except Exception:
        pass
    return w

def plot_eigs_by_dim(ckpts, out_dir, models_dir, data_path, zoom_xlim=(0.9, 1.0)):
    ensure_outdir(out_dir)
    device = device_auto()

    dims = sorted(set(d for _,_,d in ckpts))
    for d in dims:
        ckpts_d = [(p,k,dd) for (p,k,dd) in ckpts if dd == d]
        if not ckpts_d:
            continue

        tags_d = [f"{k}_d{d}" for _,k,_ in ckpts_d]
        cmap = plt.cm.get_cmap("tab20", max(20, len(tags_d)))
        color_map = {tag: cmap(i % cmap.N) for i, tag in enumerate(tags_d)}

        eigs_by_tag = {}
        all_imag = []
        for _, kind, _ in ckpts_d:
            tag = f"{kind}_d{d}"
            try:
                w = compute_or_load_eigs_for_model(kind, d, out_dir, models_dir, data_path, device)
                eigs_by_tag[tag] = w
                all_imag.extend(np.imag(w).tolist())
            except Exception as e:
                print(f"[warn] skip eigs for {tag}: {e}")

        if not eigs_by_tag:
            print(f"[warn] no eigenvalues to plot for d={d}")
            continue

        # tight symmetric ylim around imag parts
        if len(all_imag) == 0:
            y_min, y_max = -0.05, 0.05
        else:
            y_min = np.min(all_imag)
            y_max = np.max(all_imag)
            lim = max(abs(y_min), abs(y_max))
            lim = max(lim * 1.05, 0.02)
            y_min, y_max = -lim, lim

        # Full-plane context
        fig_full, ax_full = plt.subplots(figsize=(7, 7))
        t = np.linspace(0, 2*np.pi, 400)
        ax_full.plot(np.cos(t), np.sin(t), "--", lw=1, color="gray", alpha=0.6, label="unit circle")
        for tag, w in eigs_by_tag.items():
            ax_full.scatter(w.real, w.imag, s=22, color=color_map[tag], alpha=0.95, label=tag)
        ax_full.set_xlabel("Re(μ)"); ax_full.set_ylabel("Im(μ)")
        ax_full.set_aspect("equal", adjustable="box")
        ax_full.grid(True, alpha=0.25)
        handles, labels = ax_full.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax_full.legend(uniq.values(), uniq.keys(), fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig_full.tight_layout(rect=[0,0,0.80,1])
        out_full = os.path.join(out_dir, f"eigs_d{d}_full.png")
        fig_full.savefig(out_full, dpi=170); plt.close(fig_full)
        print(f"[ok] saved {out_full}")

        # Zoom near 1 on real axis
        fig_zoom, ax_zoom = plt.subplots(figsize=(7, 5))
        xx = np.linspace(zoom_xlim[0], zoom_xlim[1], 400)
        yy = np.sqrt(np.clip(1.0 - xx**2, 0, None))
        ax_zoom.plot(xx,  yy, "--", lw=1, color="gray", alpha=0.6)
        ax_zoom.plot(xx, -yy, "--", lw=1, color="gray", alpha=0.6)
        for tag, w in eigs_by_tag.items():
            ax_zoom.scatter(w.real, w.imag, s=28, color=color_map[tag], alpha=0.95, label=tag)
        ax_zoom.set_xlim(*zoom_xlim)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_xlabel("Re(μ)"); ax_zoom.set_ylabel("Im(μ)")
        ax_zoom.grid(True, alpha=0.3)
        handles, labels = ax_zoom.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax_zoom.legend(uniq.values(), uniq.keys(), fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig_zoom.tight_layout(rect=[0,0,0.80,1])
        out_zoom = os.path.join(out_dir, f"eigs_d{d}_zoom.png")
        fig_zoom.savefig(out_zoom, dpi=180); plt.close(fig_zoom)
        print(f"[ok] saved {out_zoom}")

# ========= MAIN =========
def main():
    parser = argparse.ArgumentParser(description="Per-dim comparisons of losses and eigenvalues across models.")
    parser.add_argument("--data", default=DATA_PATH, help="Path to training data .npy (for computing A if needed)")
    parser.add_argument("--bench", default=BENCH_DIR, help="Benchmark root dir")
    parser.add_argument("--models", default=MODELS_DIR, help="Models dir containing checkpoints")
    parser.add_argument("--out", default=OUT_DIR, help="Output dir for figures")
    parser.add_argument("--kinds", nargs="+", default=KINDS, help="Model kinds to include")
    parser.add_argument("--dims",  nargs="+", type=int, default=DIMS, help="Latent dims to include")
    args = parser.parse_args()

    data_path  = args.data
    bench_dir  = args.bench
    models_dir = args.models
    out_dir    = args.out
    logs_dir   = os.path.join(bench_dir, "logs")
    ensure_outdir(out_dir)

    # Filter checkpoints present on disk
    all_ckpts = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    ckpts = []
    for p in all_ckpts:
        kind, d = infer_kind_and_dim(p)
        if kind in args.kinds and (d in args.dims):
            ckpts.append((p, kind, d))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching kinds={args.kinds} dims={args.dims} in {models_dir}")

    # 1) Loss curves per dim
    plot_losses_by_dim(ckpts, logs_dir, out_dir, loss_keys=("recon","kl","koop"))

    # 2) Eigenvalues per dim (full + zoom)
    plot_eigs_by_dim(ckpts, out_dir, models_dir, data_path, zoom_xlim=(0.9, 1.0))

    # Summary of outputs
    dims_present = sorted(set(d for _,_,d in ckpts))
    print("\nDone. Outputs:")
    for d in dims_present:
        print(f"  - loss_recon_d{d}_ALL.png")
        print(f"  - loss_kl_d{d}_ALL.png")
        print(f"  - loss_koop_d{d}_ALL.png")
        print(f"  - eigs_d{d}_full.png")
        print(f"  - eigs_d{d}_zoom.png")

if __name__ == "__main__":
    main()
