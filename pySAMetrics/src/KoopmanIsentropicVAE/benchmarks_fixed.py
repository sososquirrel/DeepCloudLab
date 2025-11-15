# benchmark_loss_fixed.py
# Makes KVAE training identical to your standalone VAE+Koopman script:
# - KL in the objective: *no beta scaling* (beta does NOT multiply KL)
# - Koopman term weight: beta * 40 (inside multi_koopman_loss), gated by gamma (epoch >= epoch_switch)
# - Logging: store KL and Koop divided by beta (to match your training .pkl), but print raw per-epoch numbers

import os, math, pickle, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

# ---- Your models ----
from model import VAE                         # your VAE (same as training)
from autoencoder_simple import AE, StochasticAE  # for other kinds if you want them

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

# === EXACT MATCH WITH YOUR TRAINING LOSS DEFINITIONS ===
def vae_loss(recon_x, x, mu, logvar):
    """KL has NO beta in objective (identical to your training script)."""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.numel(x)
    total_loss = recon_loss + kl_div
    return total_loss, recon_loss, kl_div

def multi_koopman_loss(mu_t, mu_futures, steps, beta):
    """
    Koopman term uses: koop = beta * 40 * sum_step MSE( mu_t @ K^step , mu_{t+step} )
    with ridge on K fit (XtX + λI)^-1 XtY. Matches your training script.
    """
    lambda_reg = 1e-3
    X = mu_t
    Y = mu_futures[0]  # fit K on the shortest step mapping

    XtX = X.T @ X
    XtY = X.T @ Y
    I = torch.eye(XtX.shape[0], device=X.device)
    K = torch.linalg.pinv(XtX + lambda_reg * I) @ XtY

    losses = []
    for i, step in enumerate(steps):
        K_power = torch.matrix_power(K, step)
        pred = mu_t @ K_power
        loss = F.mse_loss(pred, mu_futures[i], reduction='mean')
        losses.append(loss)

    koopman_loss = beta * 40.0 * sum(losses)
    return koopman_loss, losses

# ------------- Training core -------------
def train_one(model, kind, train_loader, val_loader, *,
              num_epochs=200, epoch_switch=30,
              beta_sched=None, clip_grad=1.0, steps=(1,3,5,10), device="cpu"):
    """
    Trains one model variant with losses/logging identical to your training script.
    Returns:
      train_hist, val_hist  # lists of dicts with keys:
          total, recon, kl, koop, koop_steps, beta, gamma
      (kl and koop stored divided by beta, same as your training pickle)
    """
    use_koop = kind in ("ae_koop", "sae_koop", "kvae")
    is_vae   = kind in ("vae", "kvae")   # ONLY these use KL

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        beta  = 1.0 if beta_sched is None else float(beta_sched[epoch])  # matches your schedule feeding
        gamma = 1.0 if epoch >= epoch_switch else 0.0                    # gating identical to training

        # ===== TRAIN =====
        model.train()
        tl_total = tl_recon = tl_kl = tl_koop = 0.0
        stepwise_list = []
        n_batches = 0

        for x_t, x_future in train_loader:
            x_t, x_future = x_t.to(device), x_future.to(device)
            opt.zero_grad()

            # Forward at t
            recon_x, code_t, mu_t, logvar_t = parse_outputs(model(x_t))
            # KL exactly as in training (no beta scaling in objective)
            total, recon_loss, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t) if is_vae else \
                                        (F.mse_loss(recon_x, x_t, reduction='mean'), F.mse_loss(recon_x, x_t, reduction='mean'), torch.tensor(0.0, device=device))

            # Futures' codes (use mu if present to mirror training; fallback to code if AE/SAE)
            mu_futures = []
            for i in range(x_future.shape[1]):
                _, code_f, mu_f, _ = parse_outputs(model(x_future[:, i]))
                mu_futures.append(mu_f if (mu_f is not None) else code_f)

            # Koopman term (present for *_koop kinds)
            base_mu = mu_t if (mu_t is not None) else code_t
            koop_loss, step_losses = multi_koopman_loss(base_mu, mu_futures, steps, beta) if use_koop else (torch.tensor(0.0, device=device), [torch.tensor(0.0, device=device) for _ in steps])

            total = total + gamma * koop_loss
            total.backward()
            clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            tl_total += total.item()
            tl_recon += recon_loss.item()
            tl_kl    += kl_div.item()
            tl_koop  += koop_loss.item()
            stepwise_list.append([l.item() for l in step_losses])
            n_batches += 1

        # Averages
        tr_total = tl_total / n_batches
        tr_recon = tl_recon / n_batches
        tr_kl_raw   = tl_kl  / n_batches
        tr_koop_raw = tl_koop/ n_batches
        tr_steps = np.mean(stepwise_list, axis=0).tolist()

        # === LOGGING IDENTICAL TO YOUR TRAINING PICKLE ===
        train_hist.append({
            "total": tr_total,
            "recon": tr_recon,
            "kl":    (tr_kl_raw / beta) if beta > 0 else tr_kl_raw,     # divide by beta in stored logs
            "koop":  (tr_koop_raw / beta) if beta > 0 else tr_koop_raw, # divide by beta in stored logs
            "koop_steps": tr_steps,
            "beta":  beta,
            "gamma": gamma,
        })

        # ===== VALID =====
        model.eval()
        vl_total = vl_recon = vl_kl = vl_koop = 0.0
        v_stepwise = []
        n_batches = 0
        with torch.no_grad():
            for x_t, x_future in val_loader:
                x_t, x_future = x_t.to(device), x_future.to(device)

                recon_x, code_t, mu_t, logvar_t = parse_outputs(model(x_t))
                total, recon_loss, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t) if is_vae else \
                                            (F.mse_loss(recon_x, x_t, reduction='mean'), F.mse_loss(recon_x, x_t, reduction='mean'), torch.tensor(0.0, device=device))

                mu_futures = []
                for i in range(x_future.shape[1]):
                    _, code_f, mu_f, _ = parse_outputs(model(x_future[:, i]))
                    mu_futures.append(mu_f if (mu_f is not None) else code_f)

                base_mu = mu_t if (mu_t is not None) else code_t
                koop_loss, step_losses = multi_koopman_loss(base_mu, mu_futures, steps, beta) if use_koop else (torch.tensor(0.0, device=device), [torch.tensor(0.0, device=device) for _ in steps])
                total = total + gamma * koop_loss

                vl_total += total.item()
                vl_recon += recon_loss.item()
                vl_kl    += kl_div.item()
                vl_koop  += koop_loss.item()
                v_stepwise.append([l.item() for l in step_losses])
                n_batches += 1

        va_total = vl_total / n_batches
        va_recon = vl_recon / n_batches
        va_kl_raw   = vl_kl  / n_batches
        va_koop_raw = vl_koop/ n_batches
        va_steps = np.mean(v_stepwise, axis=0).tolist()

        val_hist.append({
            "total": va_total,
            "recon": va_recon,
            "kl":    (va_kl_raw / beta) if beta > 0 else va_kl_raw,       # divide by beta in stored logs
            "koop":  (va_koop_raw / beta) if beta > 0 else va_koop_raw,   # divide by beta in stored logs
            "koop_steps": va_steps,
            "beta":  beta,
            "gamma": gamma,
        })

        # --- Console prints identical to your training (use RAW values for KL & Koop) ---
        tr_steps_str = " ".join([f"K{s}: {v:.4f}" for s, v in zip(steps, tr_steps)])
        va_steps_str = " ".join([f"K{s}: {v:.4f}" for s, v in zip(steps, va_steps)])
        print(f"Epoch {epoch+1:03d} | TRAIN LOSS: {tr_total:.4f} | Recon: {tr_recon:.4f} | KL: {tr_kl_raw:.4f} | "
              f"{tr_steps_str} | Koop: {tr_koop_raw:.4f} | β: {beta:.4f}")
        print(f"Epoch {epoch+1:03d} | VAL   LOSS: {va_total:.4f} | Recon: {va_recon:.4f} | KL: {va_kl_raw:.4f} | "
              f"{va_steps_str} | Koop: {va_koop_raw:.4f} | β: {beta:.4f}")

    return train_hist, val_hist

# ------------- Latent extraction & eigen analysis -------------
@torch.no_grad()
def collect_latents(model, loader, device):
    zs = []
    for x_t, _ in loader:
        x_t = x_t.to(device)
        _, code, mu, _ = parse_outputs(model(x_t))
        zs.append((mu if mu is not None else code).detach().cpu().numpy())
    return np.concatenate(zs, axis=0)

def fit_linear_A(mu_seq_list, ridge=1e-6):
    # mu_seq_list: list of arrays of shape (T, d)
    Zt, Ztp1 = [], []
    for mu_seq in mu_seq_list:
        if mu_seq.ndim == 2 and mu_seq.shape[0] >= 2:
            Zt.append(mu_seq[:-1]); Ztp1.append(mu_seq[1:])
        else:
            raise ValueError("Each mu_seq must be (T,d) with T>=2")
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

def plot_latent_pca(Z, outpath, title="Latent PCA"):
    from sklearn.decomposition import PCA
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

# ------------- Main benchmark -------------
def run_benchmark(data_path, out_dir, dims=(4,8,16), kinds=("vae", "kvae"),
                  num_epochs=200, epoch_switch=30, steps=(1,3,5,10), seed=42, dt=1.0):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
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

    # === β schedule identical to your training ===
    beta_sched = np.clip(np.linspace(0.01, 0.5, num_epochs), 0, 1.0)

    results = {}  # nested dict: results[(kind, d)] = {...}

    for d in dims:
        for kind in kinds:
            print("\n=== Starting", kind, "latent_dim", d, "===\n")
            # --- Build model ---
            if kind in ("ae", "ae_koop"):
                model = AE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            elif kind in ("sae", "sae_koop"):
                model = StochasticAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)
            else:  # "vae", "kvae"  -> must mirror your training (VAE)
                model = VAE(input_dim=input_dim, hidden_dim=512, latent_dim=d).to(device)

            # --- Train ---
            train_hist, val_hist = train_one(
                model, kind, train_loader, val_loader,
                num_epochs=num_epochs, epoch_switch=epoch_switch,
                beta_sched=beta_sched, clip_grad=1.0, steps=steps, device=device
            )

            # --- Save checkpoint ---
            tag = f"{kind}_d{d}"
            ckpt_path = os.path.join(out_dir, "models", f"{tag}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # --- Save per-epoch losses NOW (match your training: store divided-by-beta values) ---
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

            # --- Latents (val), Koopman A, eigs (diagnostic) ---
            Z = collect_latents(model, DataLoader(MultiStepTemporalDataset(val_tensor, steps=(1,)), batch_size=256),
                                device)
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

            print(f"✓ {tag}: test MSE={mse:.6e}  MAE={mae:.6e}  ρ={eig_info['rho']:.3f}  stable%={100*eig_info['p_stable']:.1f}%")

    with open(os.path.join(out_dir, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Quick comparison table text dump
    print("\n=== Summary (test) ===")
    for (k,d) in results:
        r = results[(k,d)]
        print(f"{k:8s} d={d:2d} | MSE={r['test_mse']:.3e} MAE={r['test_mae']:.3e}  ρ={r['eig_summary']['rho']:.3f} stable%={100*r['eig_summary']['p_stable']:.1f}%")

# ------------- Entrypoint -------------
if __name__ == "__main__":
    run_benchmark(
        data_path="/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy",
        out_dir="/Volumes/LaCie/000_POSTDOC_2025/long_high_res/benchmarks_fixed",
        dims=[16],
        kinds=("vae", "kvae"),     #("ae", "sae", "ae_koop", "sae_koop"),          # same models as your training scenario
        num_epochs=200,
        epoch_switch=30,
        steps=(1, 3, 5, 10),
        seed=42,
        dt=1.0,
    )
