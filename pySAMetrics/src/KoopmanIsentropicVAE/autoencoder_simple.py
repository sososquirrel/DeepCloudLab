import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    """
    Deterministic Autoencoder matching the architecture style of your VAE:
    - Same encoder/decoder stack (Linear -> BN -> ReLU -> Dropout, etc.)
    - No stochastic sampling; latent is z = f(x)
    - forward(x) -> (recon_x, z)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.3):
        super().__init__()
        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_z = nn.Linear(hidden_dim // 2, latent_dim)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Optional: small init for last layer to keep early recon stable
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc_z(h)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z



class StochasticAE(nn.Module):
    """
    Stochastic Autoencoder (no KL in the loss).
    - Same backbone as your VAE.
    - Encoder outputs (mu, logvar); we sample z = mu + eps*exp(0.5*logvar).
    - Train with recon loss only.
    - forward(x) -> (recon_x, z, mu, logvar)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.3, min_logvar=-8.0, max_logvar=8.0):
        super().__init__()
        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Clamp range for numerical stability
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Optional gentle init for last layer
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(self.min_logvar, self.max_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar
