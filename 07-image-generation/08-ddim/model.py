import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DDPMConfig:
    in_channels: int = 1
    out_channels: int = 1
    init_channels: int = 16
    time_dim_base: int = 32
    time_dim: int = 64
    depth: int = 2
    dropout: float = 0.1
    max_len: int = 1000
    embed_dim: int = 64
    num_heads: int = 2

    timesteps: int = 1000
    beta: float = 0.005
    beta_1: float = 1e-4
    beta_T: float = 0.02
    scheduler: str = "linear"

class DDPMScheduler:
    def __init__(self, config):
        T = config.timesteps
        self.T = T

        if config.scheduler == "constant":
            beta = config.beta * np.ones(T)
            self.name = f"constant-{config.beta}-{T}"
        elif config.scheduler == "linear":
            beta = np.linspace(config.beta_1, config.beta_T, self.T)
            self.name = f"linear-{config.beta_1}-{config.beta_T}-{T}"
        else:
            raise ValueError(f"Unknow scheduler {config.scheduler}")

        self.alphas = 1 - beta
        self.alphas_bar = self.alphas.cumprod(axis=0)
        alphas_bar_prev = np.append(1.0, self.alphas_bar[:-1])
        self.sigma_q_square = beta * (1.0 - alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.sigma_q = np.sqrt(self.sigma_q_square)

    def get_alpha(self, t: int|np.ndarray|torch.Tensor) -> float|np.ndarray|torch.Tensor:
        return self.alphas[t-1]

    def get_alpha_bar(self, t: int) -> float:
        assert 1 <= t and t <= self.T
        return self.alphas_bar[:t]

    def add_noise(self, x0: torch.Tensor, eps0: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x0.ndim==2:
            alphas_bar = torch.from_numpy(self.alphas_bar).to(x0.device).type(torch.float32)[timesteps.squeeze()].view(x0.size(0),1)
        else:
            alphas_bar = torch.from_numpy(self.alphas_bar).to(x0.device).type(torch.float32)[timesteps.squeeze()].view(x0.size(0),1,1,1)
        x_t = alphas_bar.sqrt() * x0 + (1 - alphas_bar).sqrt() * eps0
        return x_t

    def plot(self, show: bool = False):
        for name, v in zip(
                ["alpha", "alpha_bar", "sigma_q_square", "sigma_q"],
                [self.alphas, self.alphas_bar, self.sigma_q_square, self.sigma_q]):
            plt.plot(v)
            plt.title(name)
            if show: plt.show()
            plt.close()

class PositionalEncoding(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        max_len = config.max_len

        P = torch.zeros((1, max_len, dim))
        pos = torch.arange(max_len, dtype=torch.float32).view(-1,1)
        div_term = torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim).view(1,-1)

        P[:, :, 0::2] = torch.sin(pos / div_term)
        P[:, :, 1::2] = torch.cos(pos / div_term)

        self.register_buffer("P", P)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, t):
        return self.P[0, t.squeeze(-1)]

class SelfAttention(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.W_q = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_k = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_o = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,C,H,W = x.size()
        q = self.W_q(x).contiguous().view(B, self.num_heads, self.head_dim, H*W).transpose(-2,-1) # (B, num_heads, HxW, head_dim)
        k = self.W_k(x).contiguous().view(B, self.num_heads, self.head_dim, H*W).transpose(-2,-1) # (B, num_heads, HxW, head_dim)
        v = self.W_v(x).contiguous().view(B, self.num_heads, self.head_dim, H*W).transpose(-2,-1) # (B, num_heads, HxW, head_dim)

        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)
        attn_scores = F.softmax(attn_scores, dim=-1)

        out = self.dropout(attn_scores) @ v # (B, num_heads, T, head_dim)
        out = out.transpose(-2, -1).contiguous().view(B, self.embed_dim, H, W) # (B, embed_dim, H, W)
        return self.W_o(out) # (B, in_channels, H, W)

class Block(nn.Module):
    def __init__(self, config, in_c, out_c, attn=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_c),
            nn.GroupNorm(out_c//4, out_c),
            nn.ReLU(),
        )
        self.time_proj = nn.Linear(config.time_dim, out_c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_c),
            nn.GroupNorm(out_c//4, out_c),
            nn.ReLU(),
        )
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1)
        if attn:
            self.attn = SelfAttention(config, out_c)

    def forward(self, x, global_timesteps):
        h = self.conv1(x)
        h = h+ self.time_proj(global_timesteps).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        h = h + self.shortcut(x) if hasattr(self, "shortcut") else x
        if hasattr(self, "attn"):
            h = self.attn(h)
        return h

class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        init_channels = self.config.init_channels
        in_channels = config.in_channels
        out_channels = config.out_channels
        depth = config.depth

        self.down_blocks = nn.ModuleList(
            [Block(config, in_channels, init_channels)] +
            [Block(config, init_channels * 2**i, init_channels * 2**(i+1)) for i in range(depth-1)]
        )
        self.downscale = nn.MaxPool2d(2)

        self.bottle_neck = Block(config, init_channels * 2**(depth-1), init_channels * 2**(depth), True)

        self.upscale = nn.Upsample(scale_factor=2)
        self.up_blocks = nn.ModuleList(
            [Block(config,
                   init_channels * 2**i + init_channels * 2**(i-1),
                   init_channels * 2**(i-1))
             for i in range(depth,0,-1)]
        )

        self.conv_out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Conv2d(in_channels=init_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x, global_timesteps):
        skip_connections = []
        for block in self.down_blocks:
            x = block(x, global_timesteps)
            skip_connections.append(x)
            x = self.downscale(x)

        x = self.bottle_neck(x, global_timesteps)

        for block in self.up_blocks:
            x = self.upscale(x)
            x = torch.concat((x, skip_connections.pop()), dim=1)
            x = block(x, global_timesteps)

        x = self.conv_out(x)

        return x

class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scheduler = DDPMScheduler(config)

        self.time_mlp = nn.Sequential(
            PositionalEncoding(config, config.time_dim_base),
            nn.Linear(config.time_dim_base, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.unet= Unet(config)

    def compute_loss(self, out, noise):
        return F.mse_loss(out, noise)

    def forward(self, x, timesteps, targets=None):
        global_timesteps = self.time_mlp(timesteps)
        out = self.unet(x, global_timesteps)

        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss



def ddpm_inference(model, xT: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        xt = xT
        B = xt.shape[0]
        for t in reversed(range(model.scheduler.T)):
            alpha_t = model.scheduler.alphas[t]
            alpha_bar_t = model.scheduler.alphas_bar[t]
            sigma_q_t = model.scheduler.sigma_q[t]
            timesteps = torch.full((B,), t, device=xt.device)
            eps_hat, _ = model(xt, timesteps)
            coef = (1 - alpha_t) / math.sqrt(1 - alpha_bar_t)
            z = torch.randn(xt.shape, device=xt.device)
            xt = 1 / math.sqrt(alpha_t) * (xt - coef * eps_hat) + sigma_q_t * z

        return xt


def ddim_inference(model, xT: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
    step_ratio = model.scheduler.T // num_steps
    timesteps_seq = list(range(0, model.scheduler.T, step_ratio))
    with torch.no_grad():
        xt = xT
        B = xt.shape[0]
        for i in reversed(range(len(timesteps_seq))):
            t = timesteps_seq[i]
            t_prev = timesteps_seq[i-1] if i>0 else 0
            alpha_bar_t = model.scheduler.alphas_bar[t]
            alpha_bar_t_prev = model.scheduler.alphas_bar[t_prev] if t > 0 else 1.0
            sigma_q_t = model.scheduler.sigma_q[t]
            timesteps = torch.full((B,), t, device=xt.device)
            eps_hat, _ = model(xt, timesteps)
            pred_x0 = (xt - math.sqrt(1 - alpha_bar_t) * eps_hat) / math.sqrt(alpha_bar_t)
            dir_xt = math.sqrt(1 - alpha_bar_t_prev) * eps_hat
            z = torch.randn(xt.shape, device=xt.device)
            xt = math.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_q_t * z
        return xt


class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scheduler = DDPMScheduler(config)

        self.time_mlp = nn.Sequential(
            PositionalEncoding(config, config.time_dim_base),
            nn.Linear(config.time_dim_base, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.unet= Unet(config)

    def compute_loss(self, out, noise):
        return F.mse_loss(out, noise)

    def forward(self, x, timesteps, targets=None):
        global_timesteps = self.time_mlp(timesteps)
        out = self.unet(x, global_timesteps)

        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss

    def inference(self, xT: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xt = xT
            B = xt.shape[0]
            for t in reversed(range(self.scheduler.T)):
                alpha_t = self.scheduler.alphas[t]
                alpha_bar_t = self.scheduler.alphas_bar[t]
                sigma_q_t = self.scheduler.sigma_q[t]
                timesteps = torch.full((B,), t, device=xt.device)
                eps_hat, _ = self(xt, timesteps)
                coef = (1 - alpha_t) / math.sqrt(1 - alpha_bar_t)
                z = torch.randn(xt.shape, device=xt.device)
                xt = 1 / math.sqrt(alpha_t) * (xt - coef * eps_hat) + sigma_q_t * z

            return xt

class SpiralMLP(DDPM):
    def __init__(self, config):
        super().__init__(config)
        self.time_mlp = nn.Sequential(
            PositionalEncoding(config, config.time_dim_base),
            nn.Linear(config.time_dim_base, config.time_dim),
            nn.ReLU(),
        )

        hidden_dim = 128
        self.layer1 = nn.Linear(2, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 2)

        self.t_proj1 = nn.Linear(config.time_dim, hidden_dim)
        self.t_proj2 = nn.Linear(config.time_dim, hidden_dim)
        self.t_proj3 = nn.Linear(config.time_dim, hidden_dim)

    def forward(self, x, timesteps, targets=None):
        global_t = self.time_mlp(timesteps)
        h = self.layer1(x) + self.t_proj1(global_t)
        h = F.relu(h)
        h = self.layer2(h) + self.t_proj2(global_t)
        h = F.relu(h)
        h = self.layer3(h) + self.t_proj3(global_t)
        h = F.relu(h)
        out = self.out_layer(h)

        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss

if __name__ == "__main__":
    # ddpm = DDPM(DDPMConfig())
    # x0 = torch.randn((6, 1, 28,28))
    # eps0 = torch.randn(x0.size(), device=x0.device)
    # timesteps = torch.randint(low=0, high=ddpm.scheduler.T, size=(x0.size(0),1), device=x0.device).type(torch.long)
    # xt = ddpm.scheduler.add_noise(x0, eps0, timesteps)
    # out, _ = ddpm(xt, timesteps)
    # print("input:", x0.shape)
    # print("timesteps", timesteps.shape)
    # print("output:", out.shape)
    # print(ddpm)
    #
    # for scheduler in [DDPMScheduler(DDPMConfig()), DDPMScheduler(DDPMConfig(timesteps=100, beta_1=1e-3, beta_T=0.2))]:
    #     scheduler.plot()

    # x0_hat = ddpm.inference(torch.randn((4,1,28,28)))
    # print("x0_hat:", x0_hat.shape)

    from data import MnistConfig, Mnist, show_images
    # scheduler = DDPMScheduler(DDPMConfig())
    scheduler =  DDPMScheduler(DDPMConfig(timesteps=1000, beta_1=1e-6, beta_T=2e-6))
    # scheduler.plot(True)

    # x0,_ = next(iter(Mnist(MnistConfig(train_batch_size=36)).train_dataloader))
    # eps0 = torch.randn(x0.size(), device=x0.device)
    # timesteps = torch.arange(0, scheduler.T, device=x0.device).type(torch.long).unsqueeze(0)
    # timesteps = torch.repeat_interleave(timesteps, x0.size(0), 0)
    # for t in range(timesteps.shape[1]):
    #     xT = scheduler.add_noise(x0, eps0, timesteps[:, t].unsqueeze(-1))
    #     # show_images(x0, nrow=8)
    #     show_images(xT, nrow=8)

    x0,_ = next(iter(Mnist(MnistConfig(train_batch_size=36)).train_dataloader))
    eps0 = torch.randn(x0.size(), device=x0.device)
    timesteps = torch.tensor(scheduler.T-1, device=x0.device).type(torch.long).unsqueeze(0)
    timesteps = torch.repeat_interleave(timesteps, x0.size(0), 0)
    xT = scheduler.add_noise(x0, eps0, timesteps)
    show_images(xT, nrow=8)
