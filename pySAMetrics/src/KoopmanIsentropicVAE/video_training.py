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

# ====== CREATE OUTPUT DIRECTORY ======
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ====== GENERATE AND SAVE FIGURES ======
n_total = 200
N=350
for n_time in list(range(1, n_total + 1)) + [N]:
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))

    # Total Loss
    axs[0].plot(train_losses[:n_time], label='Train Total')
    axs[0].plot(val_losses[:n_time], label='Val Total')
    axs[0].set_title("Total Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Koopman Loss
    axs[1].plot(train_koopman_losses[:n_time], label='Train Koopman')
    axs[1].plot(val_koopman_losses[:n_time], label='Val Koopman')
    axs[1].set_title("Koopman Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    # KL Divergence
    axs[2].plot(train_kl_losses[:n_time], label='Train KL')
    axs[2].plot(val_kl_losses[:n_time], label='Val KL')
    axs[2].set_title("KL Divergence")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    # Reconstruction Loss
    axs[3].plot(train_recon_losses[:n_time], label='Train Recon')
    axs[3].plot(val_recon_losses[:n_time], label='Val Recon')
    axs[3].set_title("Reconstruction Loss")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")
    axs[3].legend()

    fig.suptitle(f"Training Progress up to Epoch {n_time}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"losses_epoch_{n_time:03d}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    if n_time % 50 == 0 or n_time == 1:
        print(f"✅ Saved: {fig_path}")

print("✅✅ All 400 figures generated and saved.")
