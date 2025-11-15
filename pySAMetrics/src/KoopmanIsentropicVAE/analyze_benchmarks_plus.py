# analyze_models.py
import os, re, glob, json, csv, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# removed t-SNE usage per request

# ========= USER CONFIG =========
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
#BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks"
BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks_fixed"
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUT_DIR    = os.path.join(BENCH_DIR, "analysis_comp")
os.makedirs(OUT_DIR, exist_ok=True)

# compare these kinds (order controls plot ordering)
KINDS = ["ae", "ae_koop", "sae", "sae_koop", "vae", "kvae",
         "betavae", "wae", "betatcvae", "residualvae", "spectralvae"]
# compare these latent dims
DIMS  = [4, 8, 16]  # change to [8] or [8,16,32] etc.

# heavy ops toggles
DECODE_MODES     = True    # decode eigen-modes (needs model.decode/decoder)
SHOW_RECON_PANEL = True    # export recon panels
DT = 1.0                   # sampling interval for continuous eig conversion

# Optional path for coloring PCA/TSNE by organization index (length N array)
ORG_INDEX_PATH = os.path.join(BENCH_DIR, "organization_index.npy")
# Path for organization index based on data_evolution_pw
PW_INDEX_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/var_pw.npy'

# ========= GRID / TRANSFORMS (adjust to your data) =========
#nx, ny = 48, 48

# ========== PARAMETERS ==========
path_input = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/reshaped_rho_w_sum.npy'
# ========== LOAD AND CENTER DATA (for vis back-transform) ==========
plot_data = np.load(path_input)  # Shape: (T, 48, 48)
mean_data = plot_data.mean(axis=0)   # Center each pixel over time

# --- Helper Function: Convert Flat Tensor to Image ---
valid_indices_path = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/valid_indices.npy'
valid_indices = np.load(valid_indices_path)

def create_image_from_flat_tensor(x, valid_indices=valid_indices):
    images = np.zeros((x.shape[0], 48*48))
    images[:, valid_indices] = x
    images = images.reshape(-1, 48, 48)
    return images

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

def try_load_org_index(N_total):
    if os.path.exists(ORG_INDEX_PATH):
        arr = np.load(ORG_INDEX_PATH)
        if len(arr) >= N_total:
            return arr
    return None

def try_load_pw_index(N_total):
    """Load organization index from data_evolution_pw and crop to match data length"""
    if os.path.exists(PW_INDEX_PATH):
        arr = np.load(PW_INDEX_PATH)
        if len(arr) >= N_total:
            return arr[:N_total]  # Crop to match data length
    return None

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
    steps = int(max(10, round(steps)))
    lam = lambda_i / np.abs(lambda_i) if unit_circle else lambda_i  # normalize to unit circle if requested
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
train_np  = data[:train_end].copy()
val_np    = data[train_end:val_end].copy()
test_np   = data[val_end:].copy()
train_tensor = torch.tensor(train_np, dtype=torch.float32)
val_tensor   = torch.tensor(val_np, dtype=torch.float32)
test_tensor  = torch.tensor(test_np, dtype=torch.float32)
org_index = try_load_org_index(N)
pw_index = try_load_pw_index(N)

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

def fit_linear_A_from_Z(Z, ridge=1e-6, center=True, return_center=True):
    """
    Regress Z_{t+1} on Z_t **after removing the time mean** (anomalies).
    Returns A, eigvals, eigvecs, and the time-mean (if requested).
    """
    if center:
        Z_mean = Z.mean(axis=0, keepdims=True)
        Zc = Z - Z_mean
    else:
        Z_mean = np.zeros((1, Z.shape[1]), dtype=Z.dtype)
        Zc = Z

    X, Y = Zc[:-1], Zc[1:]                       # anomalies only
    d = Z.shape[1]
    A = np.linalg.pinv(X.T @ X + ridge*np.eye(d)) @ (X.T @ Y)   # (d,d)
    w, V = np.linalg.eig(A)
    if return_center:
        return A, w, V, Z_mean.squeeze()
    return A, w, V

def calculate_explained_variance(Z, V):
    """Explained variance of each eigenvector using **centered** latents."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    total_var = np.sum(np.linalg.norm(Zc, axis=1) ** 2)
    explained_variance = []

    for i in range(V.shape[1]):
        v_i = V[:, i]
        if np.iscomplexobj(v_i):
            # treat real/imag projections
            pR = Zc @ v_i.real
            pI = Zc @ v_i.imag
            mode_power = np.sum(pR**2) + np.sum(pI**2)
        else:
            p = Zc @ v_i
            mode_power = np.sum(p**2)
        explained_variance.append(mode_power / (total_var + 1e-12))
    return np.array(explained_variance)

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

def remove_mean(x):
    # remove per-sample spatial mean across features
    return x - x.mean(axis=1, keepdims=True)

def plot_training_curves(tag, losses_path, out_dir):
    try:
        with open(losses_path, "rb") as f:
            logs = pickle.load(f)
        train_hist = logs.get("train_hist", [])
        val_hist   = logs.get("val_hist", [])
        # per-model quick export (kept simple)
        keys = ["recon", "kl", "koop", "koop_diag"]
        for k in keys:
            if len(train_hist)==0 or k not in train_hist[0]:
                continue
            plt.figure(figsize=(6,4))
            plt.plot([e[k] for e in train_hist], label="train")
            if len(val_hist)>0 and k in val_hist[0]:
                plt.plot([e[k] for e in val_hist], label="val")
            plt.title(f"{tag} · {k}")
            plt.xlabel("epoch"); plt.ylabel(k)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_curve_{k}.png"), dpi=140)
            plt.close()
    except Exception as e:
        print(f"  (skip curves) {e}")

def load_training_logs(losses_path):
    try:
        with open(losses_path, "rb") as f:
            logs = pickle.load(f)
        return logs.get("train_hist", []), logs.get("val_hist", [])
    except Exception:
        return [], []

def plot_comparison_curves(all_logs_by_dim_kind, keys, out_dir):
    # fixed color per model kind - user specified colors
    kind_palette = {
        "ae": "#87CEEB",  # light blue
        "ae_koop": "#000080",  # dark blue  
        "sae": "#FFC0CB",  # pink
        "sae_koop": "#800080",  # purple
        "vae": "#FFA500",  # orange
        "kvae": "#FF0000",  # red
        "betavae": "#e377c2", "wae": "#7f7f7f",
        "betatcvae": "#bcbd22", "residualvae": "#17becf", "spectralvae": "#393b79",
    }
    for d, kind_to_logs in all_logs_by_dim_kind.items():
        for k in keys:
            plt.figure(figsize=(8, 5))
            has_any = False
            for kind, logs in kind_to_logs.items():
                train_hist, val_hist = logs
                first_train_keys = train_hist[0].keys() if len(train_hist)>0 else {}
                first_val_keys = val_hist[0].keys() if len(val_hist)>0 else {}
                color = kind_palette.get(kind, None)
                # special-case for Koopman: plot 0 if not present
                if (k not in first_train_keys) and (k not in first_val_keys):
                    if k == "koop":
                        plt.plot([0, 1], [0, 0], label=f"{kind} · train (0)", color=color, alpha=0.9)
                        has_any = True
                    # skip otherwise
                    continue
                if len(train_hist)>0 and k in first_train_keys:
                    plt.plot([e[k] for e in train_hist], label=f"{kind} · train", alpha=0.9, color=color)
                    has_any = True
                if len(val_hist)>0 and k in first_val_keys:
                    plt.plot([e[k] for e in val_hist], label=f"{kind} · val", ls="--", alpha=0.9, color=color)
                    has_any = True
            if not has_any:
                plt.close()
                continue
            title = f"d={d} · {k}"
            if k == "koop":
                title += " (actual Koopman loss; 0 for AE/VAE)"
            if k == "koop_diag":
                title += " (diagnostic)"
            plt.title(title)
            plt.xlabel("epoch"); plt.ylabel(k)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"comp_curves_{k}_d{d}.png"), dpi=150)
            plt.close()

def pca_scatter(Z, title, outpath, colors=None):
    pca = PCA(n_components=2).fit(Z)
    Z2 = pca.transform(Z)
    plt.figure(figsize=(5.5,5))
    if colors is None:
        plt.scatter(Z2[:,0], Z2[:,1], s=6, alpha=0.7)
    else:
        sc = plt.scatter(Z2[:,0], Z2[:,1], c=colors, s=6, cmap="viridis", alpha=0.85)
        plt.colorbar(sc, label="organization index", aspect=110, shrink=0.6)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

# t-SNE removed per request

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

def save_eigenvalues_table(tag, w, out_dir):
    rows = []
    for mu in w:
        rows.append([
            f"{mu.real:.5f}",
            f"{mu.imag:.5f}",
            f"{np.abs(mu):.5f}",
            f"{(2*np.pi)/np.absolute(np.angle(mu)):.2f}",
        ])
    headers = ["Re(μ)", "Im(μ)", "|μ|", "freq (cycles/step)"]
    fig, ax = plt.subplots(figsize=(6, min(0.35*len(rows)+1.5, 12)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(f"{tag} eigenvalues")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{tag}_eigenvalues_table.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

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
all_logs_by_dim_kind = {d: {} for d in DIMS}
eigs_by_dim_kind = {d: {} for d in DIMS}

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

    # --- Training curves ---
    logs_tag = f"{kind}_d{d}"
    losses_path = os.path.join(BENCH_DIR, "logs", f"{logs_tag}_losses.pkl")
    plot_training_curves(tag, losses_path, OUT_DIR)
    # collect logs for later comparative panels
    all_logs_by_dim_kind[d][kind] = load_training_logs(losses_path)

    # --- Reconstruction (TEST) raw and mean-removed ---
    recons = forward_reconstruct(model, test_tensor, device)
    err = recons - test_np
    mse = float((err**2).mean())
    mae = float(np.abs(err).mean())
    # mean-removed
    test_np_zm = remove_mean(test_np)
    recons_zm  = remove_mean(recons)
    err_zm = recons_zm - test_np_zm
    mse_zm = float((err_zm**2).mean())
    mae_zm = float(np.abs(err_zm).mean())

    # --- Latents (TRAIN) -> Koopman A & eigs
    Z_train = collect_latents(model, train_tensor, device)   # (T_train, d)
    A, w, V, Z0 = fit_linear_A_from_Z(Z_train, ridge=1e-6, center=True, return_center=True)
    lam, growth, freq_hz = discrete_to_continuous(w, dt=DT)
    rho = float(np.max(np.abs(w)))
    p_stable = float((np.abs(w) < 1.0).mean())
    eigs_by_dim_kind[d][kind] = w

    # --- PCA & t-SNE on TRAIN (colored by PW or ORG index if present)
    if pw_index is not None:
        org_train = pw_index[:train_end]
        org_val   = pw_index[train_end:val_end]
        org_test  = pw_index[val_end:]
    elif org_index is not None:
        org_train = org_index[:train_end]
        org_val   = org_index[train_end:val_end]
        org_test  = org_index[val_end:]
    else:
        org_train = org_val = org_test = None

    pca_scatter(Z_train, f"{tag} PCA (train)",
                os.path.join(OUT_DIR, f"{tag}_pca_train.png"),
                colors=org_train)
    # t-SNE removed per request

    # --- Calculate Explained Variance ---
    explained_var = calculate_explained_variance(Z_train, V)

    # --- Eigenvalue Phase Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
    scatter = ax.scatter(w.real, w.imag, s=100, c=explained_var,
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', aspect=110, shrink=0.6)
    cbar.set_label('Explained Variance')
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title(f'{tag} Eigenvalue Phase Plot')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    eig_phase_path = os.path.join(OUT_DIR, f"{tag}_eigenvalue_phase.png")
    fig.savefig(eig_phase_path, dpi=150)
    plt.close(fig)

    # Zoomed eigenvalue plot near unit circle (0.9..1 window)
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
    ax.scatter(w.real, w.imag, s=60, c=explained_var, cmap='viridis', alpha=0.85, edgecolors='black', linewidth=0.3)
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imag part')
    ax.set_title(f'{tag} eigenvalues (zoom)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_eigenvalue_phase_zoom.png"), dpi=150)
    plt.close(fig)

    # eigenvalues table image
    try:
        save_eigenvalues_table(tag, w, OUT_DIR)
    except Exception as e:
        print("  (skip eigenvalue table)", e)

    # --- Print Orbit Times for Complex Modes ---
    cidx = complex_mode_indices(w)
    print(f"  Orbit times for complex modes:")
    for i in cidx:
        lam_i = w[i]
        if np.imag(lam_i) != 0:
            period = 2 * np.pi / np.arctan2(np.imag(lam_i), np.real(lam_i))
            print(f"    Mode {i}: {period:.2f} time steps (λ = {lam_i:.3f})")

    # --- Orbits from complex modes: z_t = Re/Im( (λ^t) v ) + mean ---
    Z0 = Z_train.mean(axis=0)
    if len(cidx) == 0:
        print("  no complex eigenvalues; skip eigen-orbits")
    else:
        # Create orbits for TRAIN PCA plots only
        pca_split = PCA(n_components=2).fit(Z_train)
        Z2_split = pca_split.transform(Z_train)
        for i in cidx:
            v_i = V[:, i]                       # complex eigenvector (d,)
            v_i = v_i / (np.linalg.norm(v_i) + 1e-12)
            sigma_i = mode_rms_std(Z_train, v_i, center=Z0)
            scale_i = 2.0 * sigma_i
            # approx steps from angle, ensure int
            ang = np.angle(w[i])
            steps_orbit = 400 if abs(ang) < 1e-6 else max(50, int(round(2*np.pi/abs(ang))))
            trajR, trajI = make_orbit(w[i], v_i, Z0, steps=steps_orbit,
                                      scale=scale_i, unit_circle=True)
            Z2_R = pca_split.transform(trajR)
            Z2_I = pca_split.transform(trajI)
            fig, ax = plt.subplots(figsize=(6, 5))
            if org_train is not None:
                sc = ax.scatter(Z2_split[:,0], Z2_split[:,1], c=org_train,
                                s=6, cmap="viridis", alpha=0.7)
                plt.colorbar(sc, ax=ax, label="Organization Index")
            else:
                ax.scatter(Z2_split[:,0], Z2_split[:,1], s=6, alpha=0.7, label="TRAIN data")
            ax.plot(Z2_R[:,0], Z2_R[:,1], lw=2.0, label=f"mode {i} · Re(λ^t v)")
            ax.plot(Z2_I[:,0], Z2_I[:,1], lw=2.0, ls="--", label=f"mode {i} · Im(λ^t v)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            mag = np.abs(w[i]); ang = np.angle(w[i])
            title_bits = f"|λ|={mag:.3f}, arg={ang:.3f} rad (unit circle)"
            ax.set_title(f"{tag} eigen-orbit (mode {i}) - TRAIN: {title_bits}")
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            out_orbit = os.path.join(OUT_DIR, f"{tag}_orbit_mode{i}_train_pca.png")
            fig.savefig(out_orbit, dpi=160)
            plt.close(fig)

        # Also save oscillation stacks for complex modes (latent-orbit decoded frames)
        try:
            dev = next(model.parameters()).device
            has_decode = hasattr(model, "decode") or hasattr(model, "decoder")
            if has_decode:
                for i in cidx:
                    v_i = V[:, i]
                    v_i = v_i / (np.linalg.norm(v_i) + 1e-12)
                    sigma_i = mode_rms_std(Z_train, v_i, center=Z0)
                    scale_i = 2.0 * sigma_i
                    steps_vid = 60
                    frames = []
                    with torch.no_grad():
                        for t in range(steps_vid):
                            angle = 2*np.pi * (t/steps_vid)
                            z_t = Z0 + scale_i * (np.cos(angle)*v_i.real - np.sin(angle)*v_i.imag)
                            z_t = torch.tensor(z_t, dtype=torch.float32, device=dev)[None, :]
                            x_t = (model.decode(z_t) if hasattr(model, "decode") else model.decoder(z_t)).cpu().numpy()[0]
                            frames.append(x_t)
                    np.save(os.path.join(OUT_DIR, f"{tag}_oscillation_mode{i}.npy"), np.stack(frames, axis=0))
        except Exception as e:
            print("  (skip oscillation stacks)", e)

    # --- Reconstructed Modes for dim=8 (optional demo) ---
    if DECODE_MODES and d == 8 and kind in ("ae_koop", "vae", "kvae"):
        try:
            print(f"  Creating reconstructed modes for {tag} (dim={d})...")
            mean_latent = Z0.copy()
            dev = next(model.parameters()).device
            sigma_list = []
            for i in range(V.shape[1]):
                v_i = V[:, i]
                sigma_i = mode_rms_std(Z_train, v_i, center=mean_latent)
                sigma_list.append(sigma_i)

            reconstructed_modes = []
            with torch.no_grad():
                for i in range(V.shape[1]):
                    eigvec = V[:, i] * 2 * sigma_list[i]
                    if np.iscomplexobj(eigvec):
                        z_real = torch.tensor(mean_latent + np.real(eigvec), dtype=torch.float32, device=dev)[None, :]
                        z_imag = torch.tensor(mean_latent + np.imag(eigvec), dtype=torch.float32, device=dev)[None, :]
                        real_recon = (model.decode(z_real) if hasattr(model,"decode") else model.decoder(z_real)).cpu().numpy()[0]
                        imag_recon = (model.decode(z_imag) if hasattr(model,"decode") else model.decoder(z_imag)).cpu().numpy()[0]
                        reconstructed_modes.append(real_recon)
                        reconstructed_modes.append(imag_recon)
                    else:
                        z_in = torch.tensor(mean_latent + eigvec, dtype=torch.float32, device=dev)[None, :]
                        recon = (model.decode(z_in) if hasattr(model,"decode") else model.decoder(z_in)).cpu().numpy()[0]
                        reconstructed_modes.append(recon)

            # Plot reconstructed modes (2 rows, up to 8 columns)
            num_cols = min(8, max(1, int(np.ceil(len(reconstructed_modes)/2))))
            fig, ax = plt.subplots(2, num_cols, figsize=(2.5*num_cols, 6))
            vmin, vmax = -40, 40
            x = np.linspace(0, 1, 48)
            z_array = np.loadtxt('z_array.txt')
            y = z_array[:48] / 1000
            XX, YY = np.meshgrid(x, y)
            levels = np.concatenate([
                np.linspace(-150, -50, 3),
                np.linspace(-50, -10, 20),
                np.linspace(-10, 10, 50),
                np.linspace(10, 50, 20),
                np.linspace(50, 150, 3)
            ])
            levels = np.sort(np.unique(levels))

            for i in range(min(len(reconstructed_modes), 16)):
                inv_log = inv_log_signed(reconstructed_modes[i])
                images = create_image_from_flat_tensor(inv_log[None, :])
                img = images[0]
                row = 0 if i % 2 == 0 else 1
                col = i // 2
                ax[row, col].contourf(XX, YY, img, cmap='RdBu_r', levels=levels, vmin=vmin, vmax=vmax)
                ax[row, col].set_title(f"mode {col} {'Re' if row==0 else 'Im'}")
                ax[row, col].grid(True)

            from matplotlib.cm import ScalarMappable
            import matplotlib.colors as mcolors
            cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap='RdBu_r'); sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                         label=r'Isentropic Mass Flux [kg$\cdot$m/s]')
            plt.tight_layout(rect=[0, 0.1, 1, 1])
            fig.savefig(os.path.join(OUT_DIR, f"{tag}_reconstructed_modes.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> {tag}_reconstructed_modes.png")
        except Exception as e:
            print(f"  (skip reconstructed modes for {tag})", e)

    # Collect summary row
    row = {
        "tag": tag, "kind": kind, "latent_dim": d,
        "test_mse": mse, "test_mae": mae,
        "rho": rho, "p_stable": p_stable,
        "growth_max": float(growth.max()),
        "A_path": os.path.join(OUT_DIR, f"{tag}_A.npy"),
        "eigs_path": eig_phase_path,
        "pca_train_path": os.path.join(OUT_DIR, f"{tag}_pca_train.png"),
        "tsne_train_path": os.path.join(OUT_DIR, f"{tag}_tsne_train.png"),
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

# Comparative training curves per dim (beta removed per request)
try:
    plot_comparison_curves(all_logs_by_dim_kind,
                           keys=["recon", "kl", "koop", "koop_diag"], out_dir=OUT_DIR)
except Exception as e:
    print("(skip comparison curves)", e)

# Combined eigenvalue scatter for d=8 (all models, color per kind)
if 8 in eigs_by_dim_kind and len(eigs_by_dim_kind[8])>0:
    kind_palette = {
        "ae": "#87CEEB",  # light blue
        "ae_koop": "#000080",  # dark blue  
        "sae": "#FFC0CB",  # pink
        "sae_koop": "#800080",  # purple
        "vae": "#FFA500",  # orange
        "kvae": "#FF0000",  # red
        "betavae": "#e377c2", "wae": "#7f7f7f",
        "betatcvae": "#bcbd22", "residualvae": "#17becf", "spectralvae": "#393b79",
    }
    fig, ax = plt.subplots(figsize=(7,7))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4)
    for kind, wvals in eigs_by_dim_kind[8].items():
        c = kind_palette.get(kind, None)
        ax.scatter(wvals.real, wvals.imag, s=45, alpha=0.9, label=kind, color=c)
    ax.set_aspect('equal')
    ax.set_xlabel('Real part'); ax.set_ylabel('Imag part')
    ax.set_title('Eigenvalues · d=8')
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'eigs_all_kinds_d8.png'), dpi=150)
    plt.close(fig)
    
    # Zoomed eigenvalue scatter for d=8 (near unit circle)
    fig, ax = plt.subplots(figsize=(6,6))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4)
    for kind, wvals in eigs_by_dim_kind[8].items():
        c = kind_palette.get(kind, None)
        ax.scatter(wvals.real, wvals.imag, s=45, alpha=0.9, label=kind, color=c)
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Real part'); ax.set_ylabel('Imag part')
    ax.set_title('Eigenvalues · d=8 (zoom)')
    ax.legend(fontsize=8, ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'eigs_all_kinds_d8_zoom.png'), dpi=150)
    plt.close(fig)

# ========= INDIVIDUAL RECONSTRUCTION PANELS FOR EACH MODEL =========
print("\n▶ Creating individual reconstruction panels for each model...")

# Random samples for reconstruction comparison (same for all models)
np.random.seed(42)  # For reproducible random samples
Nsamp = min(5, len(test_np))
samples = np.random.choice(len(test_np), size=Nsamp, replace=False)
print("Random samples:", samples)

# Create grid for contour plots (use 48x48 to match create_image_from_flat_tensor output)
x = np.linspace(0, 1, 48)
z_array = np.loadtxt('z_array.txt')
y = z_array[:48] / 1000
XX, YY = np.meshgrid(x, y)

# Define contour levels exactly like your example
vmin, vmax = -40, 40
levels = np.concatenate([
    np.linspace(-150, -50, 3),
    np.linspace(-50, -10, 20),
    np.linspace(-10, 10, 50),
    np.linspace(10, 50, 20),
    np.linspace(50, 150, 3)
])
levels = np.sort(np.unique(levels))

# Create individual reconstruction panels for each model
for mpath, kind, d in ckpts:
    tag = os.path.splitext(os.path.basename(mpath))[0]
    print(f"  Creating reconstruction panel for {tag}...")

    # Load model
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

    # Get reconstructions
    recons = forward_reconstruct(model, test_tensor, device)

    figsamp, axs = plt.subplots(2, len(samples), figsize=(30, 19.2))  # 2 rows, Nsamp cols
    figsamp.subplots_adjust(wspace=0.3, hspace=-0.3)

    cb_plot = None
    for i, idx in enumerate(samples):
        inv_log = inv_log_signed(test_np[idx])
        images = create_image_from_flat_tensor(inv_log[None, :])
        origi_img = images[0] + mean_data

        inv_log = inv_log_signed(recons[idx])
        images = create_image_from_flat_tensor(inv_log[None, :])
        recon_img = images[0] + mean_data

        cb_plot = axs[0, i].contourf(XX, YY, origi_img, cmap='RdBu_r',
                                     levels=levels, vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f"Original #{idx}")
        axs[0, i].grid(True)

        cb_plot = axs[1, i].contourf(XX, YY, recon_img, cmap='RdBu_r',
                                     levels=levels, vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f"Reconstruction #{idx}")
        axs[1, i].grid(True)

    if cb_plot is not None:
        cbar = figsamp.colorbar(cb_plot, ax=axs.ravel().tolist(),
                                orientation='horizontal', shrink=0.8, pad=-0.5, aspect=110)
        cbar.set_label(r'Isentropic Mass Flux [kg$\cdot$m/s]')

    figsamp.suptitle(f"{tag} Reconstruction Skills", fontsize=16, y=0.98)
    figsamp.tight_layout()
    figsamp.savefig(os.path.join(OUT_DIR, f"{tag}_reconstruction_skills.png"),
                    dpi=150, bbox_inches='tight')
    plt.close(figsamp)
    print(f"    -> {tag}_reconstruction_skills.png")

    # Difference (prediction - test) panel to highlight model errors
    figerr, axerr = plt.subplots(1, len(samples), figsize=(30, 8.65))
    figerr.subplots_adjust(wspace=0.3, hspace=0.2)
    cb_plot2 = None
    for i, idx in enumerate(samples):
        inv_log_o = inv_log_signed(test_np[idx])
        inv_log_r = inv_log_signed(recons[idx])
        img_o = create_image_from_flat_tensor(inv_log_o[None, :])[0] + mean_data
        img_r = create_image_from_flat_tensor(inv_log_r[None, :])[0] + mean_data
        diff_img = img_r - img_o
        cb_plot2 = axerr[i].contourf(XX, YY, diff_img, cmap='RdBu_r', levels=levels, vmin=vmin, vmax=vmax)
        axerr[i].set_title(f"Error #{idx} (recon - orig)")
        axerr[i].grid(True)
    if cb_plot2 is not None:
        cbar2 = figerr.colorbar(cb_plot2, ax=axerr.ravel().tolist(), orientation='horizontal', shrink=0.8, pad=0.05, aspect=110)
        cbar2.set_label(r'Isentropic Mass Flux error [kg$\cdot$m/s]')
    figerr.suptitle(f"{tag} Reconstruction Errors", fontsize=16, y=0.98)
    figerr.tight_layout()
    figerr.savefig(os.path.join(OUT_DIR, f"{tag}_reconstruction_errors.png"), dpi=150, bbox_inches='tight')
    plt.close(figerr)
    print(f"    -> {tag}_reconstruction_errors.png")

print("\nDone. You now have per-model figures and per-dim comparison charts + CSV in:", OUT_DIR)
