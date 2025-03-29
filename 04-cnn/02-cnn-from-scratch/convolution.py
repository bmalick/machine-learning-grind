import torch
import torch.nn as nn

def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """2D cross-correlation computation"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Conv2DScratch(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=="max":
                Y[i,j] = X[i:i + p_h, j:j + p_w].max()
            elif mode=="avg":
                Y[i,j] = X[i:i + p_h, j:j + p_w].mean()
    return Y
