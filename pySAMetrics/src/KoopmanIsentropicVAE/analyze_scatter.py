#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scatter_kl_koop_color_recon.py

For each latent dimension, plot a scatter of models:
  x = final KL loss
  y = final Koopman loss
  color = final reconstruction loss
(using final entries from val_hist if available, else train_hist)

Expects logs at: {BENCH_DIR}/logs/{kind}_d{dim}_losses.pkl
"""

import os, re, glob, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt

# ========= DEFAULTS (you can override via CLI) =========
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"  # unused here; kept for consistency
BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks_fixed"
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUT_DIR    = os.path.join(BENCH_DIR, "analysis_comp")

KINDS = ["ae", "ae_koop", "sae", "sae_koop", "vae", "kvae",
         "betavae", "wae", "betatcvae", "residualvae", "spectralvae"]
DIMS  = [4, 8, 16]

# ========= HELPERS =========
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

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

def _last_with_key(hist, key_opts):
    """Return last entry value for any key in key_opts (in priority order)."""
    if not hist:
        return None
    # scan from end for non-None
    for e in reversed(hist):
        for k in key_opts:
            if k in e and e[k] is not None:
                return e[k]
    return None

def extract_final_losses(logs, prefer_val=True):
    """
    Return (recon, kl, koop) as floats or None.
    Priority: val_hist last non-None -> train_hist last non-None
    KL/Koop priority of keys: *_raw first, then base keys.
    """
    if logs is None:
        return None, None, None

    # choose hist
    val_hist = logs.get("val_hist", [])
    train_hist = logs.get("train_hist", [])
    hist = val_hist if (prefer_val and len(val_hist)) else train_hist

    # recon is usually present
    recon = _last_with_key(hist, ["recon"])

    # prefer *_raw if present (from your fixed benchmark); else fallback to plain
    kl   = _last_with_key(hist, ["kl_raw", "kl"])
    koop = _last_with_key(hist, ["koop_raw", "koop"])

    # if kl/koop entirely missing, treat as 0 so we can still place AE etc.
    if kl is None:
        kl = 0.0
    if koop is None:
        koop = 0.0

    # convert to float if they come as tensors/np types
    def _to_float(x):
        if x is None: return None
        try:
            return float(x)
        except Exception:
            return float(np.asarray(x).squeeze())

    return _to_float(recon), _to_float(kl), _to_float(koop)

def scatter_by_dim(ckpts, logs_dir, out_dir, annotate=True, cmap_name="viridis"):
    ensure_outdir(out_dir)

    dims = sorted(set(d for _,_,d in ckpts))
    for d in dims:
        ckpts_d = [(p,k,dd) for (p,k,dd) in ckpts if dd == d]
        if not ckpts_d:
            continue

        tags, X, Y, C = [], [], [], []
        missing = []

        for _, kind, _ in ckpts_d:
            tag = f"{kind}_d{d}"
            logs = load_logs(kind, d, logs_dir)
            recon, kl, koop = extract_final_losses(logs, prefer_val=True)

            if recon is None:  # if recon missing, we can't color it; skip
                missing.append(tag)
                continue
            tags.append(tag); X.append(kl); Y.append(koop); C.append(recon)

        if not X:
            print(f"[warn] nothing to plot for d={d}")
            continue

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        C = np.asarray(C, dtype=float)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X, Y, c=C, s=90, cmap=cmap_name, edgecolors="k", alpha=0.95)
        cb = plt.colorbar(sc)
        cb.set_label("final reconstruction loss")

        plt.xlabel("final KL loss")
        plt.ylabel("final Koopman loss")
        plt.title(f"Final losses scatter (d={d})")
        plt.grid(True, alpha=0.25)

        if annotate:
            # small offset to reduce overlap
            for (x, y, tag) in zip(X, Y, tags):
                plt.annotate(tag, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"scatter_kl_vs_koop_color_recon_d{d}.png")
        plt.savefig(out_path, dpi=170)
        plt.close()
        print(f"[ok] saved {out_path}")
        if missing:
            print(f"  (note) skipped (no recon in logs): {', '.join(missing)}")

# ========= MAIN =========
def main():
    ap = argparse.ArgumentParser(description="Scatter KL vs Koopman, colored by Recon (final epoch), per latent dim.")
    ap.add_argument("--bench",  default=BENCH_DIR,  help="Benchmark root dir")
    ap.add_argument("--models", default=MODELS_DIR, help="Models dir containing checkpoints")
    ap.add_argument("--out",    default=OUT_DIR,    help="Output dir for figures")
    ap.add_argument("--kinds",  nargs="+", default=KINDS, help="Model kinds to include")
    ap.add_argument("--dims",   nargs="+", type=int, default=DIMS, help="Latent dims to include")
    ap.add_argument("--no-annotate", action="store_true", help="Disable point annotations")
    args = ap.parse_args()

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

    scatter_by_dim(ckpts, logs_dir, out_dir, annotate=(not args.no_annotate))

    print("\nDone. Outputs per dim:")
    for d in sorted(set(dd for _,_,dd in ckpts)):
        print(f"  - scatter_kl_vs_koop_color_recon_d{d}.png")

if __name__ == "__main__":
    main()
