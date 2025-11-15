import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

class BetaVAE(nn.Module):
    """
    β-VAE: VAE with controllable disentanglement via β parameter
    - Higher β encourages more disentangled representations
    - Good for capturing independent factors in isentropic diagrams
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=4.0, dropout_prob=0.3):
        super().__init__()
        self.beta = beta
        
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
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / torch.numel(x)
        total_loss = recon_loss + self.beta * kl_div
        return total_loss, recon_loss, kl_div


class WAE(nn.Module):
    """
    Wasserstein Autoencoder: Better reconstruction quality using MMD regularization
    - Uses Maximum Mean Discrepancy instead of KL divergence
    - Better for preserving fine details in isentropic diagrams
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, lambda_mmd=10.0, dropout_prob=0.3):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        
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
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def mmd_loss(self, z, z_prior):
        """Maximum Mean Discrepancy loss"""
        def rbf_kernel(x, y, sigma=1.0):
            x = x.unsqueeze(1)
            y = y.unsqueeze(0)
            return torch.exp(-torch.sum((x - y) ** 2, dim=-1) / (2 * sigma ** 2))
        
        k_zz = rbf_kernel(z, z)
        k_zpzp = rbf_kernel(z_prior, z_prior)
        k_zzp = rbf_kernel(z, z_prior)
        
        mmd = k_zz.mean() + k_zpzp.mean() - 2 * k_zzp.mean()
        return mmd


class BetaTCVAE(nn.Module):
    """
    β-TC-VAE: VAE with total correlation regularization
    - Explicitly penalizes total correlation in latent space
    - Good for learning independent factors in isentropic diagrams
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0, gamma=1.0, dropout_prob=0.3):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        
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
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def tc_loss(self, mu, logvar):
        """Total correlation loss"""
        # Approximate total correlation using log-ratio trick
        log_qz = -0.5 * torch.sum(logvar, dim=1)
        log_qz_prod = -0.5 * torch.sum(torch.log(torch.sum(torch.exp(logvar), dim=0) + 1e-8))
        tc = torch.mean(log_qz) - log_qz_prod
        return tc


class ResidualVAE(nn.Module):
    """
    Residual VAE: VAE with residual connections for better gradient flow
    - Helps with training deeper networks
    - Better for complex isentropic diagram patterns
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.3):
        super().__init__()
        
        # Encoder with residual blocks
        self.encoder_input = nn.Linear(input_dim, hidden_dim)
        self.encoder_res1 = self._make_residual_block(hidden_dim, dropout_prob)
        self.encoder_res2 = self._make_residual_block(hidden_dim, dropout_prob)
        self.encoder_res3 = self._make_residual_block(hidden_dim, dropout_prob)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with residual blocks
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_res1 = self._make_residual_block(hidden_dim, dropout_prob)
        self.decoder_res2 = self._make_residual_block(hidden_dim, dropout_prob)
        self.decoder_res3 = self._make_residual_block(hidden_dim, dropout_prob)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def _make_residual_block(self, dim, dropout_prob):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def _residual_forward(self, x, res_block):
        residual = x
        out = res_block(x)
        out += residual
        return F.relu(out)

    def encode(self, x):
        h = F.relu(self.encoder_input(x))
        h = self._residual_forward(h, self.encoder_res1)
        h = self._residual_forward(h, self.encoder_res2)
        h = self._residual_forward(h, self.encoder_res3)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_input(z))
        h = self._residual_forward(h, self.decoder_res1)
        h = self._residual_forward(h, self.decoder_res2)
        h = self._residual_forward(h, self.decoder_res3)
        return self.decoder_output(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


class SpectralVAE(nn.Module):
    """
    Spectral VAE: VAE with spectral normalization for training stability
    - Uses spectral normalization on weights for better training dynamics
    - Good for stable training on complex isentropic diagrams
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.3):
        super().__init__()
        
        # Encoder with spectral normalization
        self.encoder = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = spectral_norm(nn.Linear(hidden_dim // 2, latent_dim))
        self.fc_logvar = spectral_norm(nn.Linear(hidden_dim // 2, latent_dim))
        
        # Decoder with spectral normalization
        self.decoder = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, hidden_dim // 2)),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, input_dim))
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
