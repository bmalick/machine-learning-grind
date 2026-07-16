import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    embed_dim: int = 513
    num_heads: int = 8
    dropout: float = 0.1
    ff_dim: int = 2048
    num_blocks: int = 6
    max_len: int = 256
    vocab_size: int = 50257

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0

        self.head_dim = embed_dim // self.num_heads

        self.W_attn = nn.Linear(embed_dim, embed_dim*3)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        combined_proj = self.W_attn(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = combined_proj[0], combined_proj[1], combined_proj[2]
        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))

        if attn_mask is not None:
            attn_scores.masked_fill_(~attn_mask, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        out = self.dropout(attn_scores) @ v # (B, num_heads, T, head_dim)

        out = out.transpose(1,2).contiguous().view(B, T, D)
        return self.W_o(out)

class AddNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, y):
        # y comes from the sublayer
        return self.ln(x + self.dropout(y))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_1 = nn.Linear(config.embed_dim, config.embed_dim * 4)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(config.embed_dim * 4, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.relu(self.W_1(x))
        return self.dropout(self.W_2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiheadAttention(config)
        self.addnorm1 = AddNorm(config)
        self.ffn = FeedForward(config)
        self.addnorm2 = AddNorm(config)

    def forward(self, x, attn_mask=None):
        x = self.addnorm1(x, self.mha(x, attn_mask))
        x = self.addnorm2(x, self.ffn(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_len = config.max_len
        embed_dim = config.embed_dim

        P = torch.zeros((1, max_len, embed_dim))
        pos = torch.arange(max_len, dtype=torch.float32).view(-1,1)
        div_term = torch.pow(10000, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim).view(1,-1)

        P[:, :, 0::2] = torch.sin(pos / div_term)
        P[:, :, 1::2] = torch.cos(pos / div_term)

        self.register_buffer("P", P)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.P[:, :x.size(1), :]
        return self.dropout(x)

class GPTPretrained(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(config.vocab_size, config.embed_dim),
            pos_enc = PositionalEncoding(config),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),
        ))
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0., std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0., std=0.02)

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def forward(self, idx, targets=None):
        x = self.transformer.embedding(idx)
        B,T,D = x.shape
        attn_mask = torch.ones((T,T), dtype=torch.bool, device=x.device).tril(diagonal=0)
        # x = self.transformer.pos_enc(x * math.sqrt(D))
        x = self.transformer.pos_enc(x)
        for block in self.transformer.blocks:
            x = block(x, attn_mask)

        # x: (B, T, embed_dim)
        logits = self.head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, seq_len: int = 256, max_tokens: int = 512):
        self.eval()
        B,T = idx.shape
        for _ in range(max_tokens):
            idx_pad = idx[:, -seq_len:]
            logits, _ = self(idx_pad)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


if __name__ == "__main__":
    gpt = GPTPretrained(GPTConfig(embed_dim=512, num_heads=4, num_blocks=6, max_len=256, vocab_size=25000))
    bs = 4
    seq_len = 10
    vocab_size = 25000
    x = torch.randint(0, vocab_size, (bs, seq_len), dtype=torch.long)
    logits, loss = gpt(x)
    print("tokens:", x.shape)
    print("logits:", logits.shape)
    print(gpt)
