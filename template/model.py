import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModuleConfig:
    num_blocks: int = 2
    dropout: float = 0.0

class Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def compute_loss(self, y_hat, y_true):
        raise NotImplementedError

    def number_of_params():
        pass

    def forward(self, x, targets=None):
        out = self.net(x)
        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss


if __name__ == "__main__":
    model = Module(ModuleConfig())
    bs = 4
    # x = torch.randint(0, vocab_size, (bs, ), dtype=torch.long)
    out, loss = model(x)
    print("tokens:", x.shape)
    print("logits:", logits.shape)
    print(model)
