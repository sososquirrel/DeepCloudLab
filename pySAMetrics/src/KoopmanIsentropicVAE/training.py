import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from model import VAE
import matplotlib.pyplot as plt
import pickle

# ========== SETUP SEED & DEVICE ==========
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD DATA ==========
data_path = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy'
data = np.load(data_path)
input_dim = data.shape[1]

# ========== MODEL & OPTIMIZER ==========
latent_dim = 8
model = VAE(input_dim=input_dim, hidden_dim=512, latent_dim=latent_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ========== TRAINING PARAMETERS ==========
num_epochs = 400
epoch_switch = 30
clip_grad = 1.0
beta_schedule = np.clip(np.linspace(0.01, 0.5, num_epochs), 0, 1.0)
steps = [1, 3, 5, 10, 15]

# ========== DATA SPLITS ==========
n_total = len(data)
train_end = int(0.95 * n_total)
val_end = train_end + int(0.025 * n_total)

train_tensor = torch.tensor(data[:train_end], dtype=torch.float32)
val_tensor = torch.tensor(data[train_end:val_end], dtype=torch.float32)
test_tensor = torch.tensor(data[val_end:], dtype=torch.float32)


# ========== DATASET CLASS ==========
class MultiStepTemporalDataset(Dataset):
    def __init__(self, data, steps):
        self.data = data
        self.steps = steps
        self.max_step = max(steps)

    def __len__(self):
        return len(self.data) - self.max_step

    def __getitem__(self, idx):
        x_t = self.data[idx]
        x_future = [self.data[idx + s] for s in self.steps]
        return x_t, torch.stack(x_future)


train_loader = DataLoader(MultiStepTemporalDataset(train_tensor, steps), batch_size=128, shuffle=True)
val_loader = DataLoader(MultiStepTemporalDataset(val_tensor, steps), batch_size=128)
test_loader = DataLoader(MultiStepTemporalDataset(test_tensor, steps), batch_size=128)


# ========== LOSS FUNCTIONS ==========
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.numel(x)
    total_loss = recon_loss + kl_div
    return total_loss, recon_loss, kl_div


def multi_koopman_loss(mu_t, mu_futures, steps, beta):
    lambda_reg = 1e-3
    X = mu_t
    Y = mu_futures[0]

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

    koopman_loss = beta * 40 * sum(losses)
    return koopman_loss, losses


# ========== TRAINING ==========
train_losses, val_losses = [], []
train_koopman_losses, val_koopman_losses = [], []
train_koopman_step_losses, val_koopman_step_losses = [], []
train_kl_losses, val_kl_losses = [], []
train_recon_losses, val_recon_losses = [], []

for epoch in range(num_epochs):
    beta = beta_schedule[epoch]
    gamma = 1.0 if epoch >= epoch_switch else 0.0

    model.train()
    epoch_loss = epoch_recon_loss = epoch_kl_loss = epoch_koopman_error = 0.0
    epoch_koopman_stepwise = []

    for x_t, x_future in train_loader:
        x_t, x_future = x_t.to(device), x_future.to(device)
        optimizer.zero_grad()

        recon_x, mu_t, logvar_t = model(x_t)
        mu_futures = [model(x_future[:, i])[1] for i in range(x_future.shape[1])]

        loss, recon_loss, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t)
        koop_loss, step_losses = multi_koopman_loss(mu_t, mu_futures, steps, beta)
        total_loss = loss + gamma * koop_loss

        total_loss.backward()
        clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_div.item()
        epoch_koopman_error += koop_loss.item()
        epoch_koopman_stepwise.append([l.item() for l in step_losses])

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_recon_loss = epoch_recon_loss / len(train_loader)
    avg_kl_loss = epoch_kl_loss / len(train_loader)
    avg_koopman_error = epoch_koopman_error / len(train_loader)
    stepwise = np.mean(epoch_koopman_stepwise, axis=0)

    train_losses.append(avg_epoch_loss)
    train_recon_losses.append(avg_recon_loss)
    train_kl_losses.append(avg_kl_loss / beta)
    train_koopman_losses.append(avg_koopman_error / beta)
    train_koopman_step_losses.append(stepwise.tolist())

    step_losses_str = " ".join([f"K{step}: {val:.4f}" for step, val in zip(steps, stepwise)])
    print(
        f"Epoch {epoch+1:03d} | TRAIN LOSS: {avg_epoch_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL: {avg_kl_loss:.4f} | "
        f"{step_losses_str} | Koop: {avg_koopman_error:.4f} | β: {beta:.4f}"
    )

    # ========== VALIDATION ==========
    model.eval()
    val_loss = val_recon_loss = val_kl_loss = val_koopman_error = 0.0
    val_koopman_stepwise = []

    with torch.no_grad():
        for x_t, x_future in val_loader:
            x_t, x_future = x_t.to(device), x_future.to(device)
            recon_x, mu_t, logvar_t = model(x_t)
            mu_futures = [model(x_future[:, i])[1] for i in range(x_future.shape[1])]

            loss, recon_loss, kl_div = vae_loss(recon_x, x_t, mu_t, logvar_t)
            koop_loss, step_losses = multi_koopman_loss(mu_t, mu_futures, steps, beta)
            total_loss = loss + gamma * koop_loss

            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_div.item()
            val_koopman_error += koop_loss.item()
            val_koopman_stepwise.append([l.item() for l in step_losses])

        val_avg_epoch_loss = val_loss / len(val_loader)
        val_avg_recon_loss = val_recon_loss / len(val_loader)
        val_avg_kl_loss = val_kl_loss / len(val_loader)
        val_avg_koopman_error = val_koopman_error / len(val_loader)
        stepwise = np.mean(val_koopman_stepwise, axis=0)

        val_losses.append(val_avg_epoch_loss)
        val_recon_losses.append(val_avg_recon_loss)
        val_kl_losses.append(val_avg_kl_loss / beta)
        val_koopman_losses.append(val_avg_koopman_error / beta)
        val_koopman_step_losses.append(stepwise.tolist())

        step_losses_str = " ".join([f"K{step}: {val:.4f}" for step, val in zip(steps, stepwise)])
        print(
            f"Epoch {epoch+1:03d} | VAL   LOSS: {val_avg_epoch_loss:.4f} | Recon: {val_avg_recon_loss:.4f} | KL: {val_avg_kl_loss:.4f} | "
            f"{step_losses_str} | Koop: {val_avg_koopman_error:.4f} | β: {beta:.4f}"
        )

# ========== SAVE MODEL & LOSSES ==========
model_path = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/koopman_vae_model.pt"
torch.save(model.state_dict(), model_path)
print(f"\n✅ Model saved to: {model_path}")

loss_dict = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_koopman_losses": train_koopman_losses,
    "val_koopman_losses": val_koopman_losses,
    "train_kl_losses": train_kl_losses,
    "val_kl_losses": val_kl_losses,
    "train_recon_losses": train_recon_losses,
    "val_recon_losses": val_recon_losses,
    "train_koopman_steps": train_koopman_step_losses,
    "val_koopman_steps": val_koopman_step_losses,
}

losses_path = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/koopman_vae_losses.pkl"
with open(losses_path, "wb") as f:
    pickle.dump(loss_dict, f)
print(f"✅ Losses saved to: {losses_path}")
