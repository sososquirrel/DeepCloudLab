# benchmark.py
import numpy as np, random, pickle, os, math
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from model import VAE            # <- you may need to create AE (deterministic)
from autoencoder_simple import AE, StochasticAE
from additional_models import BetaVAE, WAE, BetaTCVAE, ResidualVAE, SpectralVAE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------- Utils -----------------
def seed_all(s=42):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def device_auto():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")

def parse_outputs(out):
    """
    Normalize model(x) outputs across AE / StochasticAE / VAE.
    Returns: (recon_x, code_for_koop, mu, logvar)
    - AE:            (recon, z)                 -> code=z,   mu=None, logvar=None
    - StochasticAE:  (recon, z) or (recon, mu, logvar) or (recon, z, mu, logvar)
                     -> code=mu if present else z
    - VAE:           (recon, mu, logvar) or (recon, z, mu, logvar) -> code=mu
    """
    if not isinstance(out, (tuple, list)):
        return out, out, None, None
    if len(out) == 2:                      # (recon, z)
        recon, z = out
        return recon, z, None, None
    if len(out) == 3:                      # (recon, mu, logvar)
        recon, mu, logvar = out
        return recon, mu, mu, logvar
    # len >= 4: (recon, z, mu, logvar, ...)
    recon, z, mu, logvar = out[:4]
    code = mu if mu is not None else z
    return recon, code, mu, logvar



class MultiStepTemporalDataset(Dataset):
    def __init__(self, data, steps):
        self.data = data; self.steps = steps; self.max_step = max(steps)
    def __len__(self): return len(self.data) - self.max_step
    def __getitem__(self, idx):
        x_t = self.data[idx]
        x_future = [self.data[idx + s] for s in self.steps]
        return x_t, torch.stack(x_future)

def fit_linear_A(mu_seq_list, ridge=1e-6):
    # mu_seq_list: list of arrays of shape (T, d) or a single (N, d) with pairs aligned
    Zt, Ztp1 = [], []
    for mu_seq in mu_seq_list:
        if mu_seq.ndim == 2:
            Zt.append(mu_seq[:-1]); Ztp1.append(mu_seq[1:])
        else:
            raise ValueError("mu_seq must be (T,d)")
    Zt = np.concatenate(Zt, axis=0)
    Ztp1 = np.concatenate(Ztp1, axis=0)
    G = Zt.T @ Zt + ridge*np.eye(Zt.shape[1])
    A = np.linalg.solve(G, Zt.T @ Ztp1).T
    w, V = np.linalg.eig(A)
    return A, w, V

def eig_summaries(eigs, dt):
    mag = np.abs(eigs)
    ang = np.angle(eigs)
    with np.errstate(divide='ignore', invalid='ignore'):
        decay = -np.log(mag)/dt
        freq = ang/(2*np.pi*dt)
    return {
        "rho": float(np.nanmax(mag)),
        "p_stable": float(np.mean(mag < 1.0)),
        "median_decay": float(np.nanmedian(np.real(decay))),
        "top_freqs": np.flip(np.sort(np.abs(freq)))[:3].tolist()
    }

def to_tensor(x, device): return torch.tensor(x, dtype=torch.float32, device=device)

# ----------------- Training core -----------------
def train_one(model, kind, train_loader, val_loader, *, num_epochs=200, epoch_switch=30,
              beta_sched=None, clip_grad=1.0, gamma_dyn=40.0, steps=(1,3,5,10), device="cpu"):
    """
    Trains one model variant.
    Returns:
      train_hist, val_hist  # lists of dicts with keys: recon, kl, koop, koop_diag, beta
    Notes:
      - 'sae' uses reparameterization but NO KL.
      - Koopman loss is added to the objective only for *_koop kinds (after epoch_switch),
        but 'koop_diag' is computed and logged for ALL kinds.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    # Local helper to normalize outputs across AE/SAE/VAE
    def _parse_outputs(out):
        """
        Returns: (recon_x, code_for_koop, mu, logvar)
        - AE:            (recon, z)                 -> code=z,   mu=None, logvar=None
        - StochasticAE:  (recon, z) or (recon, mu, logvar) or (recon, z, mu, logvar)
                         -> code=mu if present else z
        - VAE:           (recon, mu, logvar) or (recon, z, mu, logvar) -> code=mu
        """
        if not isinstance(out, (tuple, list)):
            return out, out, None, None
        if len(out) == 2:                      # (recon, z)
            recon, z = out
            return recon, z, None, None
        if len(out) == 3:                      # (recon, mu, logvar)
            recon, mu, logvar = out
            return recon, mu, mu, logvar
        # len >= 4: (recon, z, mu, logvar, ...)
        recon, z, mu, logvar = out[:4]
        code = mu if mu is not None else z
        return recon, code, mu, logvar

    # Use-Koopman and VAE flags
    use_koop = kind in ("ae_koop", "sae_koop", "kvae")
    is_vae   = kind in ("vae", "kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae")   # ALL VAE variants use KL

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        beta  = 1.0 if beta_sched is None else float(beta_sched[epoch])
        gamma = 1.0 if epoch >= epoch_switch else 0.0

        # ===== TRAIN =====
        model.train()
        tl_recon = tl_kl = tl_koop = tl_koop_diag = 0.0
        n_batches = 0

        for x_t, x_future in train_loader:
            x_t, x_future = x_t.to(device), x_future.to(device)
            opt.zero_grad()

            # Forward
            recon_x, code_t, mu_t, logvar_t = _parse_outputs(model(x_t))
            recon_loss = F.mse_loss(recon_x, x_t, reduction='mean')

            # Special loss handling for different VAE variants
            if is_vae and (mu_t is not None) and (logvar_t is not None):
                if kind == "betavae":
                    total, _, kl_div = model.vae_loss(recon_x, x_t, mu_t, logvar_t)
                elif kind == "wae":
                    # WAE uses MMD loss instead of KL
                    z_prior = torch.randn_like(mu_t)
                    mmd_loss = model.mmd_loss(mu_t, z_prior)
                    kl_div = torch.tensor(0.0, device=device)
                    total = recon_loss + model.lambda_mmd * mmd_loss
                elif kind == "betatcvae":
                    # β-TC-VAE uses special loss with total correlation
                    recon_loss = F.mse_loss(recon_x, x_t, reduction='mean')
                    kl_div = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) / torch.numel(x_t)
                    tc_loss = model.tc_loss(mu_t, logvar_t)
                    total = recon_loss + model.beta * kl_div + model.gamma * tc_loss
                else:
                    # Standard VAE loss
                    total, _, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t, beta)
            else:
                kl_div = torch.tensor(0.0, device=device)
                total  = recon_loss

            # ---- Koopman diagnostic (always computed) ----
            fut_codes = []
            for i in range(x_future.shape[1]):
                _, code_f, mu_f, _ = _parse_outputs(model(x_future[:, i]))
                fut_codes.append(mu_f if (mu_f is not None) else code_f)
            base_code = mu_t if (mu_t is not None) else code_t
            koop_diag, _ = multi_koopman_loss(base_code, fut_codes, steps)  # avg over steps

            # Add Koopman term to objective only for *_koop kinds
            if use_koop:
                koop_loss = koop_diag
                total = total + gamma * gamma_dyn * koop_loss
            else:
                koop_loss = torch.tensor(0.0, device=device)

            # Backprop
            total.backward()
            clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            # Logs
            tl_recon += recon_loss.item()
            tl_kl    += kl_div.item()
            tl_koop  += koop_loss.item()
            tl_koop_diag += koop_diag.item()
            n_batches += 1

        train_hist.append({
            "recon": tl_recon / n_batches,
            "kl":    tl_kl / n_batches,
            "koop":  tl_koop / n_batches,        # term used in training objective (0 if off)
            "koop_diag": tl_koop_diag / n_batches,  # always-computed diagnostic
            "beta":  beta,
        })

        # ===== VALIDATION =====
        model.eval()
        vl_recon = vl_kl = vl_koop = vl_koop_diag = 0.0
        n_batches = 0
        with torch.no_grad():
            for x_t, x_future in val_loader:
                x_t, x_future = x_t.to(device), x_future.to(device)

                recon_x, code_t, mu_t, logvar_t = _parse_outputs(model(x_t))
                recon_loss = F.mse_loss(recon_x, x_t, reduction='mean')

                if is_vae and (mu_t is not None) and (logvar_t is not None):
                    if kind == "betavae":
                        _, _, kl_div = model.vae_loss(recon_x, x_t, mu_t, logvar_t)
                    elif kind == "wae":
                        z_prior = torch.randn_like(mu_t)
                        mmd_loss = model.mmd_loss(mu_t, z_prior)
                        kl_div = torch.tensor(0.0, device=device)
                    elif kind == "betatcvae":
                        kl_div = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) / torch.numel(x_t)
                    else:
                        _, _, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t, beta)
                else:
                    kl_div = torch.tensor(0.0, device=device)

                fut_codes = []
                for i in range(x_future.shape[1]):
                    _, code_f, mu_f, _ = _parse_outputs(model(x_future[:, i]))
                    fut_codes.append(mu_f if (mu_f is not None) else code_f)
                base_code = mu_t if (mu_t is not None) else code_t
                koop_diag, _ = multi_koopman_loss(base_code, fut_codes, steps)
                koop_loss = koop_diag if use_koop else torch.tensor(0.0, device=device)

                vl_recon += recon_loss.item()
                vl_kl    += kl_div.item()
                vl_koop  += koop_loss.item()
                vl_koop_diag += koop_diag.item()
                n_batches += 1

        val_hist.append({
            "recon": vl_recon / n_batches,
            "kl":    vl_kl / n_batches,
            "koop":  vl_koop / n_batches,
            "koop_diag": vl_koop_diag / n_batches,
            "beta":  beta,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{kind}] epoch {epoch+1:03d}  "
                  f"val MSE={val_hist[-1]['recon']:.5f}  KL={val_hist[-1]['kl']:.5f}  "
                  f"Koop={val_hist[-1]['koop']:.5f}  KoopDiag={val_hist[-1]['koop_diag']:.5f}")

    return train_hist, val_hist



# reuse your losses with fixes
def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.numel(x)
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div

def multi_koopman_loss(mu_t, mu_futures, steps, lambda_reg=1e-3):
    X = mu_t; Y = mu_futures[0]
    XtX = X.T @ X
    XtY = X.T @ Y
    I = torch.eye(XtX.shape[0], device=X.device)

    # MPS-safe: use pinv instead of solve
    K = (torch.linalg.pinv(XtX + lambda_reg * I) @ XtY).T  # keep the .T to match your original orientation

    losses = []
    for i, step in enumerate(steps):
        pred = mu_t @ torch.matrix_power(K, step).T
        losses.append(F.mse_loss(pred, mu_futures[i], reduction='mean'))

    return sum(losses) / len(losses), losses

# ------------- Latent extraction & eigen analysis -------------
@torch.no_grad()
def collect_latents(model, loader, kind, device):
    zs = []
    for x_t, _ in loader:
        x_t = x_t.to(device)
        _, code, mu, _ = parse_outputs(model(x_t))
        zs.append((mu if mu is not None else code).detach().cpu().numpy())
    return np.concatenate(zs, axis=0)


def plot_latent_pca(Z, outpath, title="Latent PCA"):
    pca = PCA(n_components=2).fit(Z)
    Z2 = pca.transform(Z)
    plt.figure(); plt.scatter(Z2[:,0], Z2[:,1], s=4)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_eigs(eigs, outpath, title="Eigenvalues"):
    unit = np.exp(1j*np.linspace(0,2*np.pi,400))
    plt.figure()
    plt.plot(unit.real, unit.imag, linestyle='--')
    plt.scatter(eigs.real, eigs.imag, s=12)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('Re(λ)'); plt.ylabel('Im(λ)'); plt.title(title); plt.tight_layout()
    plt.savefig(outpath); plt.close()

# ------------- Main grid -------------
def run_benchmark(data_path, out_dir, dims=(2,4,8,12,16,32), kinds=("ae", "sae", "vae", "ae_koop", "sae_koop", "kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae"),
                  num_epochs=200, epoch_switch=30, steps=(1,3,5,10), seed=42, dt=1.0):
    os.makedirs(out_dir, exist_ok=True)
    seed_all(seed); device = device_auto()

    data = np.load(data_path)
    input_dim = data.shape[1]

    n_total = len(data)
    train_end = int(0.95 * n_total)
    val_end = train_end + int(0.025 * n_total)

    train_tensor = torch.tensor(data[:train_end], dtype=torch.float32)
    val_tensor   = torch.tensor(data[train_end:val_end], dtype=torch.float32)
    test_tensor  = torch.tensor(data[val_end:], dtype=torch.float32)

    train_loader = DataLoader(MultiStepTemporalDataset(train_tensor, steps), batch_size=128, shuffle=True)
    val_loader   = DataLoader(MultiStepTemporalDataset(val_tensor, steps),   batch_size=128)
    test_loader  = DataLoader(MultiStepTemporalDataset(test_tensor, steps),  batch_size=128)

    beta_sched = np.clip(np.linspace(0.01, 1.0, num_epochs), 0, 1.0)

    results = {}  # nested dict: results[(kind, d)] = {...}

    for d in dims:
        for kind in kinds:
            # --- Build model ---
            if kind in ("ae", "ae_koop"):
                model = AE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            elif kind in ("sae", "sae_koop"):
                model = StochasticAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            elif kind == "betavae":
                model = BetaVAE(input_dim=input_dim, hidden_dim=512, latent_dim=d, beta=4.0).to(device)
            elif kind == "wae":
                model = WAE(input_dim=input_dim, hidden_dim=512, latent_dim=d, lambda_mmd=10.0).to(device)
            elif kind == "betatcvae":
                model = BetaTCVAE(input_dim=input_dim, hidden_dim=512, latent_dim=d, beta=1.0, gamma=1.0).to(device)
            elif kind == "residualvae":
                model = ResidualVAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            elif kind == "spectralvae":
                model = SpectralVAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            else:  # "vae", "kvae"
                model = VAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)

            # --- Train (MUST unpack both) ---
            train_hist, val_hist = train_one(
                model, kind, train_loader, val_loader,
                num_epochs=num_epochs, epoch_switch=epoch_switch,
                beta_sched=beta_sched, gamma_dyn=40.0, steps=steps, device=device
            )
            assert isinstance(train_hist, list) and isinstance(val_hist, list), "train_one must return (train_hist, val_hist)"

            # --- Save checkpoint ---
            tag = f"{kind}_d{d}"
            os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
            ckpt_path = os.path.join(out_dir, "models", f"{tag}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # --- Also save per-epoch losses NOW so they exist even if later code crashes ---
            losses_path = os.path.join(out_dir, "logs", f"{tag}_losses.pkl")
            with open(losses_path, "wb") as f:
                pickle.dump({"train_hist": train_hist, "val_hist": val_hist}, f)

            # --- Evaluate on test ---
            model.eval()
            mse_sum = 0.0; mae_sum = 0.0; n = 0
            with torch.no_grad():
                for x_t, _ in test_loader:
                    x_t = x_t.to(device)
                    out = model(x_t)
                    recon_x = out[0] if isinstance(out, (tuple, list)) else out
                    mse_sum += F.mse_loss(recon_x, x_t, reduction='sum').item()
                    mae_sum += torch.abs(recon_x - x_t).sum().item()
                    n += x_t.numel()
            mse = mse_sum / n; mae = mae_sum / n

            # --- Latents (val), Koopman A, eigs ---
            Z = collect_latents(model, DataLoader(MultiStepTemporalDataset(val_tensor, steps=(1,)), batch_size=256),
                                kind, device)
            L = Z.shape[0]
            A_hat, w, V = fit_linear_A([Z[:L]], ridge=1e-6)
            eig_info = eig_summaries(w, dt=dt)

            # --- Store run result ---
            results[(kind, d)] = {
                "checkpoint": ckpt_path,
                "losses_path": losses_path,
                "train_hist": train_hist,
                "val_hist": val_hist,
                "test_mse": mse,
                "test_mae": mae,
                "A_hat": A_hat.tolist(),
                "eigs": [complex(ev) for ev in w],
                "eig_summary": eig_info,
            }

            print(f"✓ {tag}: test MSE={mse:.6e}  MAE={mae:.6e}  ρ={eig_info['rho']:.3f} p_stable={eig_info['p_stable']:.2f}")


    with open(os.path.join(out_dir, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Quick comparison table text dump
    print("\n=== Summary (test) ===")
    for (k,d) in results:
        r = results[(k,d)]
        print(f"{k:8s} d={d:2d} | MSE={r['test_mse']:.3e} MAE={r['test_mae']:.3e}  ρ={r['eig_summary']['rho']:.3f} stable%={100*r['eig_summary']['p_stable']:.1f}%")


if __name__ == "__main__":
    run_benchmark(
        data_path="/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy",
        out_dir="/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks",
        dims=(4, 8, 16, 32),
        kinds=("ae", "sae", "ae_koop", "vae"), #("kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae", "ae", "sae", "vae", "ae_koop", "sae_koop", "kvae", "betavae", "wae", "betatcvae", "residualvae", "spectralvae"),
        num_epochs=5,
        epoch_switch=30,
        steps=(1, 3, 5, 10),
        seed=42,
        dt=1.0,
    )
