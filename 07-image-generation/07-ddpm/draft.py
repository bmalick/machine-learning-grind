import numpy as np
import matplotlib.pyplot as plt
import torch

from data import Mnist, MnistConfig, show_images

class VarianceScheduler:
    def __init__(self, T: int = 1000):
        self.T = T

    def plot_alphas(self):
        plt.plot(self.alphas)
        plt.title(r"$\alpha_t$")
        plt.show()

    def plot_alphas_bar(self):
        plt.plot(self.alphas_bar)
        plt.title(r"$\bar{\alpha_t}$")
        plt.show()

    def add_noise(self, x0: torch.Tensor, eps0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alphas_bar = torch.from_numpy(self.alphas_bar)[t.squeeze()].view(x0.size(0),1,1,1)
        x_t = alphas_bar.sqrt() * x0 + (1 - alphas_bar).sqrt() * eps0
        return x_t, eps0

# print(x0.shape)


class ConstantVarianceScheduler(VarianceScheduler):
    name = "constant"
    def __init__(self, T: int = 1000, beta: float = 0.005):
        super().__init__(T=T)
        self.beta = beta

        self.alphas = 1 - np.ones(shape=(T,)) * beta
        self.alphas_bar = self.alphas.cumprod(axis=0)

class LinearVarianceScheduler(VarianceScheduler):
    name = "linear"
    def __init__(self, T: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02):
        assert beta_1 < beta_T
        super().__init__(T=T)
        self.beta_1 = beta_1
        self.beta_T = beta_T

        var_func = lambda t: (beta_T - beta_1) / (T-1) * (t-1) + beta_1
        self.alphas = 1 - np.vectorize(var_func)(np.arange(1, T+1))
        self.alphas_bar = self.alphas.cumprod(axis=0)

mnist = Mnist(MnistConfig(train_batch_size=32))
x0, _ = next(iter(mnist.train_dataloader))

# for var_scheduler in [LinearVarianceScheduler(), ConstantVarianceScheduler(beta=1e-8)]:
#     # var_scheduler.plot_alphas()
#     # var_scheduler.plot_alphas_bar()
#
#     # t = torch.randint(low=0, high=var_scheduler.T, size=(x0.size(0),1))
#     t = torch.tensor([0]*x0.size(0), dtype=torch.long)
#     eps0 = torch.randn(x0.size())
#     x_t = var_scheduler.add_noise(x0, eps0, t)
#     show_images(x_t, nrow=8)
#

import math
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_channels, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.W_q = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_k = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.W_o = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,C,H,W = x.size()
        q = self.W_q(x) # (B, embed_dim, H, W)
        k = self.W_k(x) # (B, embed_dim, H, W)
        v = self.W_v(x) # (B, embed_dim, H, W)


        q = q.contiguous().view(B, self.embed_dim, -1).transpose(1,2) # (B, HxW, embed_dim)
        k = k.contiguous().view(B, self.embed_dim, -1).transpose(1,2) # (B, HxW, embed_dim)
        v = v.contiguous().view(B, self.embed_dim, -1).transpose(1,2) # (B, HxW, embed_dim)

        B, T, D = q.shape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)


        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)
        attn_scores = F.softmax(attn_scores, dim=-1)

        out = self.dropout(attn_scores) @ v # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, embed_dim)
        out = out.view(B, -1, H, W) # (B, C, H, W)
        return self.W_o(out) # (B, in_channels, H, W)

# x = torch.randn((4, 32, 7, 7))
# attn = Attention(32, 64, 8)
# attn(x)

