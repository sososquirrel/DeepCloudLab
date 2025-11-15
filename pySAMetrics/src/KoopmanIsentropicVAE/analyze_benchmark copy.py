# analyze_models.py
import os, re, glob, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ========= USER CONFIG =========
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks"
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUT_DIR    = os.path.join(BENCH_DIR, "analysis_comp")
os.makedirs(OUT_DIR, exist_ok=True)

# compare these kinds (order controls plot ordering)
KINDS = ["ae", "ae_koop", "sae", "sae_koop", "vae", "kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae"]
# compare these latent dims
DIMS  = [8, 16]  # change to [8] or [8,16,32] etc.

# heavy ops toggles
DECODE_MODES     = False   # True to decode eigen-modes (needs model.decode/decoder)
SHOW_RECON_PANEL = False   # True to export quick recon panels
DT = 1.0                   # sampling interval for continuous eig conversion

# ========= GRID / TRANSFORMS (adjust to your data) =========
#nx, ny = 48, 48

def infer_grid(D):
    # return factor pair (nx, ny) with shape closest to square
    best = None
    for ny in range(1, int(np.sqrt(D)) + 1):
        if D % ny == 0:
            nx = D // ny
            cand = (nx, ny)
            if best is None or abs(nx - ny) < abs(best[0] - best[1]):
                best = cand
    return best



def inv_log_signed(x):  # optional pretty transform
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

# ========= MODELS =========
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
    # Normalize forward outputs across AE/SAE/VAE
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

def infer_kind_and_dim(filename):
    base = os.path.basename(filename)
    m_dim = re.search(r"_d(\d+)\.pt$", base)
    d = int(m_dim.group(1)) if m_dim else None
    kind = base.split("_d")[0]
    return kind, d

def complex_mode_indices(w, tol=1e-10):
    """Return indices of eigenvalues with positive imaginary part (one per conj pair)."""
    w = np.asarray(w)
    return [i for i in range(len(w)) if np.imag(w[i]) > tol]

def mode_rms_std(Z, v, center=None):
    """
    RMS std of the complex projection (Z - center) @ v.
    Uses sqrt( Var(Re) + Var(Im) ).
    """
    if center is None:
        center = Z.mean(axis=0)
    alpha = (Z - center) @ v  # complex series length T
    return np.sqrt(np.var(alpha.real) + np.var(alpha.imag))

def make_orbit(lambda_i, v_i, center, steps=400, scale=1.0, unit_circle=True):
    """
    Build latent orbits along eigenvector v_i:
      X0 = scale * v_i
      z_t = (λ_i^t) * X0
    Returns: (traj_real_plus_center, traj_imag_plus_center), both (steps, d) real arrays.
    """
    lam = lambda_i / np.abs(lambda_i) if unit_circle else lambda_i
    X0 = v_i * scale
    t = np.arange(steps, dtype=np.int64)
    lam_t = lam ** t[:, None]                 # (steps, 1) complex
    traj = lam_t * X0[None, :]                # (steps, d) complex
    return (traj.real + center), (traj.imag + center)  # both real (steps, d)


# ========= DATA =========
device = device_auto()
data = np.load(DATA_PATH)
D = data.shape[1]
nx, ny = infer_grid(D)
print(f"Inferred grid: nx={nx}, ny={ny}  (nx*ny={nx*ny} == D={D})")
assert nx * ny == D
assert D == nx*ny, f"nx*ny={nx*ny} must match data dim {D}"
N = len(data)
train_end = int(0.95 * N)
val_end   = train_end + int(0.025 * N)
val_np    = data[train_end:val_end].copy()
test_np   = data[val_end:].copy()
val_tensor  = torch.tensor(val_np, dtype=torch.float32)
test_tensor = torch.tensor(test_np, dtype=torch.float32)



# ========= UTILITIES =========
def forward_reconstruct(model, x_tensor, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x_tensor), batch_size=256, shuffle=False):
            xb = xb.to(device)
            out = model(xb)
            recon, _, _, _ = parse_outputs(out)
            outs.append(recon.detach().cpu())
    return torch.cat(outs, dim=0).numpy()

@torch.no_grad()
def collect_latents(model, x_tensor, device):
    Zs = []
    for (xb,) in DataLoader(TensorDataset(x_tensor), batch_size=256, shuffle=False):
        xb = xb.to(device)
        out = model(xb)
        _, code, mu, _ = parse_outputs(out)
        Zs.append((mu if mu is not None else code).detach().cpu().numpy())
    return np.concatenate(Zs, axis=0)  # (T, d)

def fit_linear_A_from_Z(Z, ridge=1e-6):
    X, Y = Z[:-1], Z[1:]
    d = Z.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    A = np.linalg.pinv(XtX + ridge * np.eye(d)) @ XtY
    w, V = np.linalg.eig(A)
    return A, w, V

def discrete_to_continuous(w, dt=1.0):
    eps = 1e-12
    lam = np.log(np.clip(np.abs(w), eps, None)) / dt + 1j * (np.angle(w) / dt)
    return lam, lam.real, lam.imag / (2*np.pi)

def cmplx_unit_circle(ax=None):
    t = np.linspace(0, 2*np.pi, 400)
    u = np.exp(1j*t)
    ax = ax or plt.gca()
    ax.plot(u.real, u.imag, "--", lw=1)
    return ax

def to_img(x_flat): return x_flat.reshape(ny, nx)

def decode_modes(model, Z_mean, V, eps=0.5, device="cpu"):
    has_decode = hasattr(model, "decode")
    has_decoder = hasattr(model, "decoder")
    if not (has_decode or has_decoder):
        raise AttributeError("Model has no .decode or .decoder")
    modes = []
    with torch.no_grad():
        for k in range(V.shape[1]):
            v = V[:, k].real
            v = v / (np.linalg.norm(v) + 1e-12)
            z_plus  = torch.tensor(Z_mean + eps*v, dtype=torch.float32, device=device)[None, :]
            z_minus = torch.tensor(Z_mean - eps*v, dtype=torch.float32, device=device)[None, :]
            x_plus  = model.decode(z_plus)  if has_decode  else model.decoder(z_plus)
            x_minus = model.decode(z_minus) if has_decode  else model.decoder(z_minus)
            diff = (x_plus - x_minus).detach().cpu().numpy()[0] / (2*eps)
            modes.append(diff)
    return np.stack(modes, axis=0)

# ========= GATHER CHECKPOINTS (filtered) =========
all_ckpts = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
ckpts = []
for p in all_ckpts:
    kind, d = infer_kind_and_dim(p)
    if kind in KINDS and (d in DIMS):
        ckpts.append((p, kind, d))
if not ckpts:
    raise FileNotFoundError(f"No checkpoints matching kinds={KINDS} dims={DIMS} in {MODELS_DIR}")

# ========= ANALYZE & COLLECT SUMMARY =========
rows = []  # for CSV/summary
by_dim = {d: [] for d in DIMS}

for mpath, kind, d in ckpts:
    tag = os.path.splitext(os.path.basename(mpath))[0]
    print(f"\n▶ Analyzing {tag}")

    ModelCls = get_model_class(kind)
    if kind == "betavae":
        model = ModelCls(input_dim=D, hidden_dim=512, latent_dim=d, beta=4.0).to(device)
    elif kind == "wae":
        model = ModelCls(input_dim=D, hidden_dim=512, latent_dim=d, lambda_mmd=10.0).to(device)
    elif kind == "betatcvae":
        model = ModelCls(input_dim=D, hidden_dim=512, latent_dim=d, beta=1.0, gamma=1.0).to(device)
    else:
        model = ModelCls(input_dim=D, hidden_dim=512, latent_dim=d).to(device)
    state = torch.load(mpath, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # --- Reconstruction (TEST)
    recons = forward_reconstruct(model, test_tensor, device)
    err = recons - test_np
    mse = float((err**2).mean())
    mae = float(np.abs(err).mean())

    # --- Latents (VAL) -> Koopman A & eigs
    Z = collect_latents(model, val_tensor, device)  # contiguous
    A, w, V = fit_linear_A_from_Z(Z, ridge=1e-6)
    lam, growth, freq_hz = discrete_to_continuous(w, dt=DT)
    rho = float(np.max(np.abs(w)))
    p_stable = float((np.abs(w) < 1.0).mean())

    # --- PCA & orbit (for figures)
    # --- PCA on Z (VAL) ---
pca = PCA(n_components=2).fit(Z)
Z2 = pca.transform(Z)

# --- Orbits from complex modes: z_t = Re/Im( (λ^t) v ) + mean ---
Z0 = Z.mean(axis=0)
steps_orbit = 400          # tweak as you like
use_unit_circle = True     # set False to include growth/decay (|λ| ≠ 1)

cidx = complex_mode_indices(w)
if len(cidx) == 0:
    print("  no complex eigenvalues; skip eigen-orbits")
else:
    for i in cidx:
        v_i = V[:, i]                       # complex eigenvector (d,)
        # normalize eigenvector to unit norm to make scale interpretable
        v_i = v_i / (np.linalg.norm(v_i) + 1e-12)

        # data-driven amplitude: 2 * RMS std of projection along v_i
        sigma_i = mode_rms_std(Z, v_i, center=Z0)
        scale_i = 2.0 * sigma_i

        # build orbits
        trajR, trajI = make_orbit(w[i], v_i, Z0, steps=steps_orbit,
                                  scale=scale_i, unit_circle=use_unit_circle)

        # project with the SAME PCA fitted on Z
        Z2_R = pca.transform(trajR)
        Z2_I = pca.transform(trajI)

        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(Z2[:,0], Z2[:,1], lw=0.8, alpha=0.35, label="VAL traj")
        ax.plot(Z2_R[:,0], Z2_R[:,1], lw=2.0, label=f"mode {i} · Re(λ^t v)")
        ax.plot(Z2_I[:,0], Z2_I[:,1], lw=2.0, ls="--", label=f"mode {i} · Im(λ^t v)")
        ax.scatter(Z2[0,0], Z2[0,1], s=25, zorder=5, label="VAL start")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        mag = np.abs(w[i]); ang = np.angle(w[i])
        title_bits = f"|λ|={mag:.3f}, arg={ang:.3f} rad"
        if use_unit_circle: title_bits += " (unit circle)"
        ax.set_title(f"{tag} eigen-orbit (mode {i}): {title_bits}")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{tag}_orbit_mode{i}_pca.png"), dpi=160)
        plt.close(fig)


    # Optional: decoded eigen-modes
    if DECODE_MODES:
        try:
            modes = decode_modes(model, Z.mean(axis=0), V, eps=0.5, device=device)
            kshow = min(d, 8)
            fig, axes = plt.subplots(1, kshow, figsize=(3*kshow, 3))
            if kshow == 1: axes = [axes]
            for i in range(kshow):
                im = inv_log_signed(modes[i])
                vmax = np.percentile(np.abs(im), 99)
                axes[i].imshow(to_img(im), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                axes[i].set_title(f"mode {i+1}"); axes[i].axis("off")
            fig.suptitle(f"{tag} decoded eigen-modes")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"{tag}_decoded_modes.png"), dpi=150); plt.close(fig)
        except Exception as e:
            print("  (skip decoded modes)", e)

    # Optional: recon panels
    if SHOW_RECON_PANEL:
        Nsamp = min(5, len(test_np))
        figsamp, axs = plt.subplots(2, Nsamp, figsize=(3.0*Nsamp, 5.5))
        for i in range(Nsamp):
            axs[0, i].imshow(inv_log_signed(test_np[i]).reshape(ny, nx), cmap="RdBu_r")
            axs[0, i].set_title(f"orig #{i}"); axs[0, i].axis("off")
            axs[1, i].imshow(inv_log_signed(recons[i]).reshape(ny, nx), cmap="RdBu_r")
            axs[1, i].set_title("recon"); axs[1, i].axis("off")
        figsamp.suptitle(f"{tag} recon (anomaly)")
        figsamp.tight_layout()
        figsamp.savefig(os.path.join(OUT_DIR, f"{tag}_recons.png"), dpi=150); plt.close(figsamp)

    # Collect summary row
    row = {
        "tag": tag, "kind": kind, "latent_dim": d,
        "test_mse": mse, "test_mae": mae,
        "rho": rho, "p_stable": p_stable,
        "growth_max": float(growth.max()),  # of slowest (largest |w|)
        "A_path": os.path.join(OUT_DIR, f"{tag}_A.npy"),
        "eigs_path": os.path.join(OUT_DIR, f"{tag}_eigs.png"),
        "pca_orbit_path": os.path.join(OUT_DIR, f"{tag}_pca_orbit.png"),
    }
    np.save(row["A_path"], A)
    rows.append(row)
    by_dim[d].append(row)

# ========= SAVE TABLE (CSV + JSONL) =========
csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT_DIR, "summary.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"\nSaved summary to {csv_path}")

# ========= MAKE COMPARISON PLOTS (per dim) =========
def bar_comp(rows_for_dim, metric, title, fname, ylabel=None, rot=30):
    kinds_order = [k for k in KINDS if any(r["kind"]==k for r in rows_for_dim)]
    vals = [next(r[metric] for r in rows_for_dim if r["kind"]==k) for k in kinds_order]
    plt.figure(figsize=(max(6, 1.5*len(kinds_order)), 4))
    plt.bar(range(len(kinds_order)), vals)
    plt.xticks(range(len(kinds_order)), kinds_order, rotation=rot)
    plt.title(title)
    plt.ylabel(ylabel or metric)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150); plt.close()
    print("  ->", out)

for d in DIMS:
    if not by_dim[d]:
        continue
    rows_d = by_dim[d]

    bar_comp(rows_d, "test_mse",
             f"Test MSE by kind (d={d})",
             f"comp_mse_d{d}.png", ylabel="MSE (lower is better)")

    bar_comp(rows_d, "test_mae",
             f"Test MAE by kind (d={d})",
             f"comp_mae_d{d}.png", ylabel="MAE (lower is better)")

    bar_comp(rows_d, "rho",
             f"Spectral radius ρ(A) by kind (d={d})",
             f"comp_rho_d{d}.png", ylabel="ρ = max|μ|")

    # Convert stability fraction to percent for plotting
    rows_pct = [{**r, "p_stable_pct": 100.0 * r["p_stable"]} for r in rows_d]
    bar_comp(rows_pct, "p_stable_pct",
             f"Stable eigenvalues (%) by kind (d={d})",
             f"comp_stable_d{d}.png", ylabel="% eigs with |μ|<1")

print("\nDone. You now have per-model figures and per-dim comparison charts + CSV in:", OUT_DIR)
