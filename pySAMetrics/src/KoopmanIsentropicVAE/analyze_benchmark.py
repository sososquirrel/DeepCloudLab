# analyze_models.py
import os, re, glob, json, csv, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# ========= USER CONFIG =========
DATA_PATH  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy"
#BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks"
BENCH_DIR  = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks_fixed"
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUT_DIR    = os.path.join(BENCH_DIR, "analysis_comp")

# Check if paths exist
if not os.path.exists(BENCH_DIR):
    print(f"ERROR: BENCH_DIR does not exist: {BENCH_DIR}")
    exit(1)
if not os.path.exists(MODELS_DIR):
    print(f"ERROR: MODELS_DIR does not exist: {MODELS_DIR}")
    exit(1)

# Create output directory
try:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
except Exception as e:
    print(f"ERROR: Could not create output directory {OUT_DIR}: {e}")
    exit(1)

# compare these kinds (order controls plot ordering)
KINDS = ["ae", "ae_koop", "sae", "sae_koop", "vae", "kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae"]
# compare these latent dims
DIMS  = [4, 8, 16]  # change to [8] or [8,16,32] etc.

# heavy ops toggles
DECODE_MODES     = True    # decode eigen-modes (needs model.decode/decoder)
SHOW_RECON_PANEL = True    # export recon panels
DT = 1.0                   # sampling interval for continuous eig conversion

# Optional path for coloring PCA by organization index (length N array)
ORG_INDEX_PATH = os.path.join(BENCH_DIR, "organization_index.npy")
# Path for organization index based on data_evolution_pw
PW_INDEX_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/var_pw.npy'

# ========= GRID / TRANSFORMS (adjust to your data) =========
#nx, ny = 48, 48

# ========== PARAMETERS ==========
path_input = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/reshaped_rho_w_sum.npy'
# ========== LOAD AND CENTER DATA ==========
data = np.load(path_input)  # Shape: (T, 48, 48)
mean_data = data.mean(axis=0)   # Center each pixel over time


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
    lam = lambda_i / np.abs(lambda_i) if unit_circle else lambda_i #no attenuation for the orbits when unit circle
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

def try_load_org_index(N_total):
    if os.path.exists(ORG_INDEX_PATH):
        arr = np.load(ORG_INDEX_PATH)
        if len(arr) >= N_total:
            return arr
    return None

def plot_training_curves(tag, losses_path, out_dir):
    try:
        with open(losses_path, "rb") as f:
            logs = pickle.load(f)
        train_hist = logs.get("train_hist", [])
        val_hist   = logs.get("val_hist", [])
        keys = ["recon", "kl", "koop", "koop_diag", "beta"]
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

def create_pca_multiplot(all_pca_data, out_dir):
    """Create multiplot of all PCA spaces with density-based coloring"""
    if not all_pca_data:
        return
    
    # Calculate grid size
    n_plots = len(all_pca_data)
    n_cols = min(4, n_plots)  # max 4 columns
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Collect all density values for consistent colorbar
    all_densities = []
    density_data = []
    
    for tag, Z2, org_colors in all_pca_data:
        # Calculate point density using Gaussian KDE
        
        # Create grid for density estimation
        x_min, x_max = Z2[:, 0].min(), Z2[:, 0].max()
        y_min, y_max = Z2[:, 1].min(), Z2[:, 1].max()
        
        # Use Gaussian KDE for smooth density estimation
        try:
            kde = gaussian_kde(Z2.T)
            density = kde(Z2.T)
        except:
            # Fallback to simple histogram-based density
            density = np.ones(len(Z2))
        
        all_densities.extend(density)
        density_data.append((tag, Z2, density))
    
    # Normalize all densities to [0, 1] for consistent colorbar
    all_densities = np.array(all_densities)
    vmin, vmax = all_densities.min(), all_densities.max()
    
    # Plot each PCA
    for i, (tag, Z2, density) in enumerate(density_data):
        if i >= len(axes):
            break
            
        # Normalize density for this plot
        density_norm = (density - vmin) / (vmax - vmin + 1e-10)
        
        sc = axes[i].scatter(Z2[:, 0], Z2[:, 1], c=density_norm, 
                           s=8, cmap='viridis', alpha=0.8)
        axes[i].set_title(tag, fontsize=10)
        axes[i].set_xlabel("PC1", fontsize=8)
        axes[i].set_ylabel("PC2", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(density_data), len(axes)):
        axes[i].set_visible(False)
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Point Density', fontsize=12)
    
    plt.suptitle('Latent Space PCA - All Models (Density Colored)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_multiplot_density.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> pca_multiplot_density.png")

def create_sae_kvae_comparison(all_pca_data, out_dir):
    """Create specific comparison plot for SAE, SAE+koop, KVAE with d4 and d8"""
    # Filter data for specific models and dimensions
    target_models = ['sae', 'sae_koop', 'kvae']
    target_dims = [4, 8]
    
    filtered_data = []
    for tag, Z2, org_colors in all_pca_data:
        # Parse tag to get model and dimension
        parts = tag.split('_d')
        if len(parts) == 2:
            model = parts[0]
            dim = int(parts[1])
            if model in target_models and dim in target_dims:
                filtered_data.append((tag, Z2, org_colors, model, dim))
    
    if len(filtered_data) == 0:
        print("  No matching data found for SAE/KVAE comparison")
        return
    
    # Organize data: 2 rows (d4, d8) x 3 cols (sae, sae_koop, kvae)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    cbar_ax = fig.add_axes([0.15, -0.1, 0.7, 0.02]) 
    
    # Collect all density values for consistent colorbar
    all_densities = []
    density_data = []
    
    for tag, Z2, org_colors, model, dim in filtered_data:
        # Calculate density
        try:
            kde = gaussian_kde(Z2.T)
            density = kde(Z2.T)
        except:
            density = np.ones(len(Z2))
        
        all_densities.extend(density)
        density_data.append((tag, Z2, density, model, dim))
    
    # Normalize all densities
    all_densities = np.array(all_densities)
    vmin, vmax = all_densities.min(), all_densities.max()
    
    # Create PCA object to get explained variance
    pca_full = PCA(n_components=2)
    
    # Plot organized by dimension and model
    for tag, Z2, density, model, dim in density_data:
        # Calculate explained variance
        pca_full.fit(Z2)
        explained_var = pca_full.explained_variance_ratio_
        
        # Determine position in grid
        row = 0 if dim == 4 else 1  # d4=row0, d8=row1
        col_map = {'sae': 0, 'sae_koop': 1, 'kvae': 2}
        col = col_map.get(model, 0)
        
        # Normalize density
        density_norm = (density - vmin) / (vmax - vmin + 1e-10)
        
        # Plot
        sc = axes[row, col].scatter(Z2[:, 0], Z2[:, 1], c=density_norm, 
                                   s=10, cmap='viridis', alpha=0.8)
        
        # Title with explained variance
        title = f"{model.upper()}, axis I: {explained_var[0]*100:.1f}%, axis II: {explained_var[1]*100:.1f}%"
        axes[row, col].set_title(title, fontsize=11)
        axes[row, col].set_xlabel("PC1", fontsize=10)
        axes[row, col].set_ylabel("PC2", fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        
        # Add dimension labels on the left
        if col == 0:
            axes[row, col].set_ylabel(f"d{dim}\nPC2", fontsize=10)
    
    # Add model labels at the top
    model_labels = ['SAE', 'SAE+Koop', 'KVAE']
    for col, label in enumerate(model_labels):
        axes[0, col].text(0.5, 1.15, label, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add shared horizontal colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', 
                       shrink=0.8, aspect=30)
    cbar.set_label('Point Density', fontsize=12)
    
    #plt.suptitle('Latent Space PCA Comparison - SAE vs SAE+Koop vs KVAE', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_sae_kvae_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> pca_sae_kvae_comparison.png")

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
all_pca_data = []  # store (tag, Z2, colors) for multiplot

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

    # --- Latents (VAL) -> Koopman A & eigs
    #Z = collect_latents(model, train_tensor, device)  # contiguous
    Z_train = collect_latents(model, train_tensor, device)   # (T_val, d)
    A, w, V, Z0 = fit_linear_A_from_Z(Z_train, ridge=1e-6, center=True, return_center=True)
    lam, growth, freq_hz = discrete_to_continuous(w, dt=DT)
    rho = float(np.max(np.abs(w)))
    p_stable = float((np.abs(w) < 1.0).mean())

    # --- PCA & orbit (for figures)
    # PCA on splits
    #Z_val   = Z
    #Z_test  = collect_latents(model, test_tensor, device)
    
    # Use PW index for coloring if available, otherwise fall back to org_index
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
    
    pca_scatter(Z_train, f"{tag} PCA (train)", os.path.join(OUT_DIR, f"{tag}_pca_train.png"), colors=org_train)
    #pca_scatter(Z_val,   f"{tag} PCA (val)",   os.path.join(OUT_DIR, f"{tag}_pca_val.png"),   colors=org_val)
    #pca_scatter(Z_test,  f"{tag} PCA (test)",  os.path.join(OUT_DIR, f"{tag}_pca_test.png"),  colors=org_test)
    
    # Store PCA data for multiplot
    pca = PCA(n_components=2).fit(Z_train)
    Z2 = pca.transform(Z_train)
    all_pca_data.append((tag, Z2, org_train))

    # --- Calculate Explained Variance ---
    explained_var = calculate_explained_variance(Z_train, V)
    
    # --- Eigenvalue Phase Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
    
    # Plot eigenvalues with explained variance as color
    scatter = ax.scatter(w.real, w.imag, s=100, c=explained_var, 
                        cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', aspect=110, shrink=0.6)
    cbar.set_label('Explained Variance')
    
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title(f'{tag} Eigenvalue Phase Plot')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_eigenvalue_phase.png"), dpi=150)
    plt.close(fig)
    
    # --- Print Orbit Times for Complex Modes ---
    cidx = complex_mode_indices(w)
    print(f"  Orbit times for complex modes:")
    for i in cidx:
        lam = w[i]
        if np.imag(lam) != 0:
            period = 2 * np.pi / np.arctan2(np.imag(lam), np.real(lam))
            print(f"    Mode {i}: {period:.2f} time steps (λ = {lam:.3f})")
    
    # --- Orbits from complex modes: z_t = Re/Im( (λ^t) v ) + mean ---
    Z0 = Z_train.mean(axis=0)
    steps_orbit = 1000          # tweak as you like should be abs(2pi/arctan(im/real))
    use_unit_circle = True     # set False to include growth/decay (|λ| ≠ 1)

    if len(cidx) == 0:
        print("  no complex eigenvalues; skip eigen-orbits")
    else:
        # Create orbits for TRAIN and TEST PCA plots only
        for split_name, Z_split, org_split in [("train", Z_train, org_train)]:
            # Fit PCA on this split
            pca_split = PCA(n_components=2).fit(Z_split)
            Z2_split = pca_split.transform(Z_split)
            
            for i in cidx:
                v_i = V[:, i]                       # complex eigenvector (d,)
                # normalize eigenvector to unit norm to make scale interpretable
                v_i = v_i / (np.linalg.norm(v_i) + 1e-12)

                # data-driven amplitude: 2 * RMS std of projection along v_i
                sigma_i = mode_rms_std(Z_train, v_i, center=Z0)
                scale_i = 2.0 * sigma_i

                steps_orbit = np.abs(2*np.pi / np.arctan(w[i].imag/w[i].real))

                # build orbits using normalized lambda (unit circle, no attenuation)
                trajR, trajI = make_orbit(w[i], v_i, Z0, steps=steps_orbit,
                                          scale=scale_i, unit_circle=True)

                # project with the PCA fitted on this split
                Z2_R = pca_split.transform(trajR)
                Z2_I = pca_split.transform(trajI)

                # plot
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # Plot data points with organization index coloring
                if org_split is not None:
                    scatter = ax.scatter(Z2_split[:,0], Z2_split[:,1], c=org_split, 
                                       s=6, cmap="viridis", alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label="Organization Index")
                else:
                    ax.scatter(Z2_split[:,0], Z2_split[:,1], s=6, alpha=0.7, label=f"{split_name.upper()} data")
                
                # Plot orbits
                ax.plot(Z2_R[:,0], Z2_R[:,1], lw=2.0, label=f"mode {i} · Re(λ^t v)")
                ax.plot(Z2_I[:,0], Z2_I[:,1], lw=2.0, ls="--", label=f"mode {i} · Im(λ^t v)")
                #ax.scatter(Z2_split[0,0], Z2_split[0,1], s=25, zorder=5, label=f"{split_name.upper()} start")
                
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                mag = np.abs(w[i]); ang = np.angle(w[i])
                title_bits = f"|λ|={mag:.3f}, arg={ang:.3f} rad"
                if use_unit_circle: title_bits += " (unit circle)"
                ax.set_title(f"{tag} eigen-orbit (mode {i}) - {split_name.upper()}: {title_bits}")
                ax.legend(loc="best", fontsize=9)
                fig.tight_layout()
                # Ensure output directory exists
                os.makedirs(OUT_DIR, exist_ok=True)
                fig.savefig(os.path.join(OUT_DIR, f"{tag}_orbit_mode{i}_{split_name}_pca.png"), dpi=160)
                plt.close(fig)


    
    # --- Reconstructed Modes for dim=8 (based on your example) ---
    if d == 8:
        try:
            print(f"  Creating reconstructed modes for {tag} (dim=8)...")
            
            # Get mean latent representation
            mean_latent = Z0.copy()
            mean_latent_t = torch.tensor(mean_latent, dtype=torch.float32,
                                     device=next(model.parameters()).device)
            
            # Calculate sigma for each mode (data-driven amplitude)
            sigma_list = []
            for i in range(V.shape[1]):
                v_i = V[:, i]
                sigma_i = mode_rms_std(Z, v_i, center=mean_latent)
                sigma_list.append(sigma_i)
            
            # Reconstruct modes
            reconstructed_modes = []
            device = next(model.parameters()).device
            
            with torch.no_grad():
                for i in range(V.shape[1]):
                    eigvec = V[:, i] * 2 * sigma_list[i]
                    
                if np.iscomplexobj(eigvec):
                    z_real = mean_latent_t + torch.tensor(np.real(eigvec), dtype=torch.float32, device=dev)
                    z_imag = mean_latent_t + torch.tensor(np.imag(eigvec), dtype=torch.float32, device=dev)
                    real_recon = model.decode(z_real.unsqueeze(0)).cpu().numpy()[0]
                    imag_recon = model.decode(z_imag.unsqueeze(0)).cpu().numpy()[0]
                    reconstructed_modes.append(real_recon)
                    reconstructed_modes.append(imag_recon)
                else:
                    z_in = mean_latent_t + torch.tensor(eigvec, dtype=torch.float32, device=dev)
                    recon = model.decode(z_in.unsqueeze(0)).cpu().numpy()[0]
                    reconstructed_modes.append(recon)
            
            # Plot reconstructed modes (2 rows, 8 columns)
            fig, ax = plt.subplots(2, 8, figsize=(20, 6))
            
            vmin, vmax = -40, 40
            
            # Create grid for contour plots
            x = np.linspace(0, 1, 48)
            z_array = np.loadtxt('z_array.txt')
            y = z_array[:48] / 1000 
            XX, YY = np.meshgrid(x, y)
            
            # Define contour levels
            levels = np.concatenate([
                np.linspace(-150, -50, 3),
                np.linspace(-50, -10, 20),
                np.linspace(-10, 10, 50),
                np.linspace(10, 50, 20),
                np.linspace(50, 150, 3)
            ])
            levels = np.sort(np.unique(levels))
            
            for i in range(len(reconstructed_modes)):
                inv_log = inv_log_signed(reconstructed_modes[i])
                images = create_image_from_flat_tensor(inv_log)
                img = images[0]
                
                row = 0 if i % 2 == 0 else 1
                col = i // 2
                
                im = ax[row, col].contourf(XX,YY,img,cmap='RdBu_r', levels=levels, vmin=vmin, vmax=vmax)
                
                # Fill empty space for real modes
                if np.imag(w[i//2]) == 0 and i % 2 == 1:
                    ax[1, col].contourf(XX,YY, np.zeros((48,48)),cmap='RdBu_r', levels=levels, vmin=vmin, vmax=vmax)
                
                ax[row, col].set_title(f"λ = {w[col]:.3f}")
            
            # Add row labels
            for row, label in zip([0, 1], ['Real part of modes 1–8', 'Imaginary part of modes 1–8']):
                ax[row, 0].set_ylabel(label, fontsize=14, labelpad=20)
            
            # Add common colorbar
            from matplotlib.cm import ScalarMappable
            import matplotlib.colors as mcolors
            cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap='RdBu_r')
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            
            plt.tight_layout(rect=[0, 0.1, 1, 1])
            fig.savefig(os.path.join(OUT_DIR, f"{tag}_reconstructed_modes.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> {tag}_reconstructed_modes.png")
            
        except Exception as e:
            print(f"  (skip reconstructed modes for {tag})", e)

    # Store reconstruction data for later individual panel creation
    # (We'll create individual panels after the main loop)

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

# ========= CREATE PCA MULTIPLOT =========
print("\n▶ Creating PCA multiplot with density coloring...")
create_pca_multiplot(all_pca_data, OUT_DIR)

print("\n▶ Creating SAE/KVAE comparison plot...")
create_sae_kvae_comparison(all_pca_data, OUT_DIR)

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

# Comprehensive reconstruction comparison removed as requested

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
    
    # Create figure exactly like your example
    #figsamp, axs = plt.subplots(2, Nsamp, figsize=(4*Nsamp, 20))
    #figsamp.subplots_adjust(wspace=0.3, hspace=-0.3)

    figsamp, axs = plt.subplots(2, len(samples), figsize=(30, 19.2))  # 2 rows, 8 columns
    figsamp.subplots_adjust(wspace=0.3, hspace=-0.3)
    
    # Create placeholder for colorbar
    cb_plot = None
    
    for i, idx in enumerate(samples):
        # Recover and reshape images (exactly like your example)
        inv_log = inv_log_signed(test_np[idx])
        images = create_image_from_flat_tensor(inv_log)
        origi_img = images[0] + mean_data

        inv_log = inv_log_signed(recons[idx])
        images = create_image_from_flat_tensor(inv_log)
        recon_img = images[0] + mean_data
        
        # Plot original
        cb_plot = axs[0, i].contourf(XX, YY, origi_img, cmap='RdBu_r',
                                    levels=levels, vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f"Original #{idx}")
        axs[0, i].grid(True)
        
        # Plot reconstructed
        cb_plot = axs[1, i].contourf(XX, YY, recon_img, cmap='RdBu_r',
                                    levels=levels, vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f"Reconstruction #{idx}")
        axs[1, i].grid(True)
    
    # Add shared horizontal colorbar (exactly like your example)
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

print("\nDone. You now have per-model figures and per-dim comparison charts + CSV in:", OUT_DIR)
