import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# ===========================================
# Approach 1: multi-line projection
# ===========================================

bs = 4              # B
seq_len = 5         # T
vocab_size = 500    # V
num_heads = 8       # H
embed_dim = 384

valid_lens = torch.tensor([1, 2, 5, 4])
# input tokens: (B, T)
X = torch.randint(0, vocab_size, size=(bs,seq_len))
print("tokens:", X.shape)
print("valid_lens:", valid_lens.shape)

embedding_layer = nn.Embedding(vocab_size, embed_dim)
X = embedding_layer(X)
print("embeddings:", X.shape) # (B, T, embed_dim)

W_q = nn.Linear(embed_dim, embed_dim)
W_k = nn.Linear(embed_dim, embed_dim)
W_v = nn.Linear(embed_dim, embed_dim)

q = W_q(X) # (B, T, embed_dim)
k = W_k(X) # (B, T, embed_dim)
v = W_v(X) # (B, T, embed_dim)
print("queries:", q.shape)
print("keys:", k.shape)
print("values:", v.shape)

# splitting
assert embed_dim % num_heads == 0
head_dim = embed_dim // num_heads
print("embed_dim:", embed_dim, "-", "num_heads:", num_heads, "-", "head_dim:", head_dim)

B, T, D = q.shape
q = q.view(B, T, num_heads, head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
k = k.view(B, T, num_heads, head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
v = v.view(B, T, num_heads, head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
print("queries split:", q.shape)
print("keys split:", k.shape)
print("values split:", v.shape)

attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)
print("scores:", attn_scores.shape)

attn_mask = torch.arange(T)[None, :] < valid_lens[:, None] # (B, T)
attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)                  # (B, 1, 1, T)
print("attn_mask:", attn_mask.shape)

attn_scores = attn_scores.masked_fill_(~attn_mask, float("-inf"))
attn_scores = F.softmax(attn_scores, dim=-1)
print("softmax scores:", attn_scores.shape)

out = attn_scores @ v # (B, num_heads, T, head_dim)
print("attention out split:", out.shape)

# concat
out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, embed_dim)
print("attention out concat:", out.shape)

W_o = nn.Linear(embed_dim, embed_dim)
out = W_o(out) # (B, T, embed_dim)
print("mha out:", out.shape)

print("="*90)

# ===================================
# Approach 2: fused linear projection
# ===================================

bs = 4              # B
seq_len = 5         # T
vocab_size = 500    # V
num_heads = 8       # H
embed_dim = 384

# input tokens: (B, T)
X = torch.randint(0, vocab_size, size=(bs,seq_len))
print("tokens:", X.shape)
print("valid_lens:", valid_lens.shape)

embedding_layer = nn.Embedding(vocab_size, embed_dim)
X = embedding_layer(X)
print("embeddings:", X.shape) # (B, T, embed_dim)

W_attn = nn.Linear(embed_dim, embed_dim * 3)

# splitting
assert embed_dim % num_heads == 0
head_dim = embed_dim // num_heads
print("embed_dim:", embed_dim, "-", "num_heads:", num_heads, "-", "head_dim:", head_dim)

B, T, D = X.shape
# (B, T, embed_dim) -> (B, T, embed_dim * 3) -> (B, T, 3, num_heads, head_dim) -> (3, B, num_heads, T, head_dim)
combined_proj = W_attn(X).view(B, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
print("combined_proj:", combined_proj.shape)

q, k, v = combined_proj[0], combined_proj[1], combined_proj[2]
print("queries split:", q.shape)
print("keys split:", k.shape)
print("values split:", v.shape)

attn_scores = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1)) # (B, num_heads, T, T)
print("scores:", attn_scores.shape)

attn_mask = torch.ones((T,T), dtype=torch.bool).tril(diagonal=0) # (T, T)
print("attn_mask:", attn_mask.shape)

attn_scores = attn_scores.masked_fill_(~attn_mask, float("-inf"))
attn_scores = F.softmax(attn_scores, dim=-1)
print("softmax scores:", attn_scores.shape)

out = attn_scores @ v # (B, num_heads, T, head_dim)
print("attention out split:", out.shape)

# concat
out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, embed_dim)
print("attention out concat:", out.shape)

W_o = nn.Linear(embed_dim, embed_dim)
out = W_o(out) # (B, T, embed_dim)
print("mha out:", out.shape)

print("="*90)

# ===================
# Positional encoding
# ===================

bs = 4              # B
seq_len = 5         # T
vocab_size = 500    # V
num_heads = 8       # H
embed_dim = 384
max_len = 1000

X = torch.randint(0, vocab_size, size=(bs,seq_len))
print("tokens:", X.shape)

embedding_layer = nn.Embedding(vocab_size, embed_dim)
X = embedding_layer(X)
print("embeddings:", X.shape) # (B, T, embed_dim)

pos_enc = torch.zeros((1, max_len, embed_dim))
pos = torch.arange(max_len, dtype=torch.float32).view(-1,1)
div_term = torch.pow(10000, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim).view(1,-1)

pos_enc[:, :, 0::2] = torch.sin(pos / div_term)
pos_enc[:, :, 1::2] = torch.cos(pos / div_term)

out = X + pos_enc[:, :X.size(1), :]
print("pos enc out:", out.shape)

print("="*90)

# ====================
# optimized dataloader
# ====================

bs = 4
text = []

with open("./TinyStories-train.txt", "r") as f:
    count = 0
    while True:
        line = f.readline()
        text.append(line)
        count += 1
        if count>bs:
            break

max_lr = 5.0e-4
min_lr = 1.0e-5
warmup_steps = 5000
max_iters = 150000
alpha = (max_lr - min_lr) / warmup_steps

def scheduler(it):
    if it < warmup_steps:
        return alpha * it + min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(it * math.pi / max_iters))

iters = np.arange(max_iters)
plt.plot(iters, [scheduler(t) for t in iters])
plt.show()

