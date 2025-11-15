import pickle
import matplotlib.pyplot as plt
import os

# ====== LOAD SAVED LOSSES ======
losses_path = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/koopman_vae_losses.pkl"

with open(losses_path, "rb") as f:
    losses = pickle.load(f)

train_losses = losses["train_losses"]
val_losses = losses["val_losses"]
train_koopman_losses = losses["train_koopman_losses"]
val_koopman_losses = losses["val_koopman_losses"]
train_kl_losses = losses["train_kl_losses"]
val_kl_losses = losses["val_kl_losses"]
train_recon_losses = losses["train_recon_losses"]
val_recon_losses = losses["val_recon_losses"]

# ====== SAVE FIGURES ======
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(22, 6))

# Total Loss
plt.subplot(1, 4, 1)
plt.plot(train_losses, label='Train Total')
plt.plot(val_losses, label='Val Total')
plt.title("Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Koopman Loss
plt.subplot(1, 4, 2)
plt.plot(train_koopman_losses, label='Train Koopman')
plt.plot(val_koopman_losses, label='Val Koopman')
plt.title("Koopman Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# KL Divergence
plt.subplot(1, 4, 3)
plt.plot(train_kl_losses, label='Train KL Divergence')
plt.plot(val_kl_losses, label='Val KL Divergence')
plt.title("KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Reconstruction Loss
plt.subplot(1, 4, 4)
plt.plot(train_recon_losses, label='Train Recon Loss')
plt.plot(val_recon_losses, label='Val Recon Loss')
plt.title("Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

# Save Figure
fig_path = os.path.join(output_dir, "training_losses.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"✅ Figure saved to: {fig_path}")
