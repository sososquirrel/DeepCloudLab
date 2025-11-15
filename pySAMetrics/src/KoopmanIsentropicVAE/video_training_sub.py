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
output_dir = "figures_2"
os.makedirs(output_dir, exist_ok=True)

n_total = 200
N=350
activation_epoch = 30  # Koopman loss activation epoch

for n_time in list(range(1, n_total + 1)) + [N]:
#for n_time in [100]:
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))
    # Flatten axs to 1D array for easier indexing
    axs = axs.flatten()

    # Row 1, Col 1: Total Loss
    axs[0].plot(train_losses[:n_time], label='Train Total')
    axs[0].plot(val_losses[:n_time], label='Val Total')
    axs[0].set_title("Total Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Row 1, Col 2: Koopman Loss (full)
    axs[1].plot(train_koopman_losses[:n_time], label='Train Koopman')
    axs[1].plot(val_koopman_losses[:n_time], label='Val Koopman')
    axs[1].set_title("Koopman Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    # Row 1, Col 3: KL Divergence
    axs[2].plot(train_kl_losses[:n_time], label='Train KL')
    axs[2].plot(val_kl_losses[:n_time], label='Val KL')
    axs[2].set_title("KL Divergence")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    # Row 2, Col 1: Reconstruction Loss
    axs[3].plot(train_recon_losses[:n_time], label='Train Recon')
    axs[3].plot(val_recon_losses[:n_time], label='Val Recon')
    axs[3].set_title("Reconstruction Loss")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")
    axs[3].legend()

    # Row 2, Col 2: Koopman Loss zoomed in (below full Koopman)
    if n_time > activation_epoch:
        start = activation_epoch
        end = min(n_time, len(train_koopman_losses))
        axs[5].plot(range(start, end), train_koopman_losses[start:end], label='Train Koopman')
        axs[5].plot(range(start, end), val_koopman_losses[start:end], label='Val Koopman')
        axs[5].set_xlim(start, n_time)
        axs[5].set_yscale('log')
        axs[5].set_title("Koopman Loss Zoomed In (Log Scale)")
        axs[5].set_xlabel("Epoch")
        axs[5].set_ylabel("Loss")
        axs[5].legend()
    else:
        axs[5].axis('off')

    # Row 2, Col 3: empty plot, hide axis
    axs[6].axis('off')
    axs[4].axis('off')
    axs[7].axis('off')
    for ax in axs:
        ax.grid(True)


    fig.suptitle(f"Training Progress up to Epoch {n_time}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"losses_epoch_{n_time:03d}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    if n_time % 50 == 0 or n_time == 1:
        print(f"✅ Saved: {fig_path}")

print("✅✅ All figures generated and saved.")

