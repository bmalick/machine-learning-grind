import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        q = self.W_q(q) # (B, T, embed_dim)
        k = self.W_k(k) # (B, T, embed_dim)
        v = self.W_v(v) # (B, T, embed_dim)


        B, T, D = q.shape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill_(~attn_mask, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        out = self.dropout(attn_scores) @ v # (B, num_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, embed_dim)
        return self.W_o(out) # (B, T, embed_dim)


class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape

        combined_proj = self.W_attn(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = combined_proj[0], combined_proj[1], combined_proj[2]

        attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill_(~attn_mask, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        out = self.dropout(attn_scores) @ v # (B, num_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, embed_dim)
        return self.W_o(out) # (B, T, embed_dim)


if __name__ == "__main__":
    bs = 4              # B
    seq_len = 5         # T
    vocab_size = 500    # V
    num_heads = 8       # H

    valid_lens = torch.tensor([1, 2, 5, 4])
    # input tokens: (B, T)
    X = torch.randint(0, vocab_size, size=(bs,seq_len))
    print("tokens:", X.shape)
    print("valid_lens:", valid_lens.shape)

    embed_dim = 384
    embedding_layer = nn.Embedding(vocab_size, embed_dim)
    X = embedding_layer(X)
    print("embeddings:", X.shape) # (B, T, embed_dim)

    print("="*90)

    B, T, D = X.shape
    padding_mask = torch.arange(T)[None, :] < valid_lens[:, None]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    print("padding_mask:", padding_mask.shape)

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    # out = mha(X, padding_mask)
    out = mha(X, X, X, padding_mask)
    print("mha out:", out.shape)

    print("="*90)

    causal_mask = torch.ones(T,T, dtype=torch.bool, device=X.device).tril(diagonal=0)
    print("causal_mask:", causal_mask.shape)
    
    causal_mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    # out = causal_mha(X, causal_mask)
    out = causal_mha(X, X, X, causal_mask)
    print("causal mha out:", out.shape)

    print("="*90)

    decoder_mask = padding_mask & causal_mask

    decoder_mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    # out = decoder_mha(X, decoder_mask)
    out = decoder_mha(X, X, X, decoder_mask)
    print("causal mha out:", out.shape)

    # print("="*90)

    # print(padding_mask)
    # print(causal_mask)
    # print(decoder_mask)
