import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModuleConfig:
    latent_dim: int = 32
    hidden_dim: int = 32
    sigma_dec: float = 1.

def gaussian_prior(latent_dim, device):
    mu = torch.zeros((1, latent_dim), device=device)
    log_var = torch.zeros((1, latent_dim), device=device)
    return (mu, log_var)

def sample(param):
    mu, log_var = param
    std = log_var.mul(0.5).exp()
    eps = torch.randn(mu.size(), device=mu.device)
    return mu + std * eps

class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.config = config
        self.sigma_dec = config.sigma_dec

    def encode(self, x): raise NotImplementedError
    def decode(self, z): raise NotImplementedError

    def log_p_guassian(self, x, p_x_given_z):
        mu_ast, log_var_ast = p_x_given_z
        var_ast = log_var_ast.exp()
        return -0.5 * ( math.log(2 * math.pi) + log_var_ast + (x.view(x.size(0), -1)-mu_ast).pow(2) / var_ast ).sum(1)


    def kl_div_gaussian(self, q_z_given_x):
        mu, log_var = q_z_given_x[0], q_z_given_x[1]
        var = log_var.exp()
        return 0.5 * (var + (mu.view(mu.size(0), -1)).pow(2) - 1 - log_var).sum(1)

    def compute_loss(self, p_x_given_z, q_z_given_x, y):
        # mu = p_x_given_z[0]
        # recon_loss = (y.flatten(1) - mu.flatten(1)).pow(2).sum(1)
        # log_p = - recon_loss / (2 * self.sigma_dec )
        log_p = self.log_p_guassian(y, p_x_given_z)
        kl_div = self.kl_div_gaussian(q_z_given_x)
        loss = (-log_p + kl_div).mean()
        return loss, log_p.mean(), kl_div.mean()

    def forward(self, x, targets=None):
        q_z_given_x = self.encode(x)
        z = sample(q_z_given_x)
        p_x_given_z = self.decode(z)
        loss, log_p, kl_div = None, None, None
        if targets is not None:
            loss, log_p, kl_div = self.compute_loss(p_x_given_z, q_z_given_x, targets)
        return p_x_given_z, (loss, log_p, kl_div)

class VariationalAutoEncoderMLP(VariationalAutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=hidden_dim), nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=2*latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim), nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=784),
            nn.Sigmoid()
        )

    def encode(self, x):
        out = self.encoder(x.flatten(1)).view(x.size(0), 2, -1)
        mu, log_var = out[:, 0], out[:, 1]
        return mu, log_var

    def decode(self, z):
        mu = self.decoder(z)
        if self.sigma_dec > 0.0:
            log_var = torch.full_like(mu, math.log(self.sigma_dec ** 2))
        else:
            log_var = torch.zeros(mu.size(), device=mu.device)
        return mu, log_var


class VariationalAutoEncoderConv(VariationalAutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=2 * latent_dim, kernel_size=4, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=4, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, kernel_size=5, stride=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # [Key note] I used view here but conv output layers are not contiguous
        out = self.encoder(x).reshape(x.size(0), 2, -1)
        mu, log_var = out[:, 0], out[:, 1]
        return mu, log_var

    def decode(self, z):
        mu = self.decoder(z.reshape(z.size(0), -1, 1, 1))
        if self.sigma_dec > 0.0:
            log_var = torch.full_like(mu, math.log(self.sigma_dec ** 2))
        else:
            log_var = torch.zeros(mu.size(), device=mu.device)
        return mu, log_var

    def kl_div_gaussian(self, q_z_given_x):
        return super().kl_div_gaussian(
                (q_z_given_x[0].flatten(1), q_z_given_x[1].flatten(1))
            )

    def log_p_guassian(self, x, p_x_given_z):
        return super().log_p_guassian(x.flatten(1), (p_x_given_z[0].flatten(1), p_x_given_z[1].flatten(1)))
