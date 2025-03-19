#!//home/malick/miniconda3/envs/pt/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def get_data(n: int, d: int, sigma: float=0.01):
    x = torch.randn(n, d)
    y = 0.05 + 0.01 * x.sum(dim=1) + torch.randn(n,1)*sigma
    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x,y),
        batch_size=5, shuffle=True
    )
    return dataloader

def l2_penalty(w):
    return (w**2).sum() / 2

dataloader = get_data(n=100, d=10)

class WeightDecayFromScrath:
    def __init__(self, num_inputs, sigma=0.01):
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def params(self): return [self.w, self.b]

    def __call__(self, x): return torch.matmul(x, self.w) + self.b



def get_criterion(lambd: float):
    def criterion(y_true, y_pred, w):
        return ((y_true-y_pred)**2 / 2).mean()+l2_penalty(w)*lambd
    return criterion

def train(lambd, criterion, optimizer, model, max_epochs=10, num_inputs = 200):
    train_dataloader = get_data(n=20, d=num_inputs)
    eval_dataloader = get_data(n=100, d=num_inputs)
    track_train_loss = []
    track_eval_loss = []
    for e in range(max_epochs):
        num_samples = 0
        epoch_loss = 0
        for batch in train_dataloader:
            if model.__class__.__name__.startswith("Concise"):
                loss = criterion(batch[-1], model(*batch[:-1]))
            else:
                loss = criterion(y_true=batch[-1], y_pred=model(*batch[:-1]), w=model.w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_samples += batch[0].size(0)
            epoch_loss += loss.item()*batch[0].size(0)
        print(f"Epoch [{e}/{max_epochs}] loss: {epoch_loss/num_samples:.3f}")
        track_train_loss.append(epoch_loss/num_samples)

        num_samples = 0
        epoch_loss = 0
        for batch in eval_dataloader:
            num_samples += batch[0].size(0)
            if model.__class__.__name__.startswith("Concise"):
                loss = criterion(batch[-1], model(*batch[:-1]))
            else:
                loss = criterion(y_true=batch[-1], y_pred=model(*batch[:-1]), w=model.w)
            epoch_loss += loss.item()*batch[0].size(0)
        track_eval_loss.append(epoch_loss/num_samples)

    plt.plot(track_train_loss, label="train_loss")
    plt.plot(track_eval_loss, label="eval_loss")
    plt.yscale("log")
    plt.title(rf"$\lambda={lambd}$")
    plt.legend(); plt.show()
    try:
        print(f"L2 norm of w for lambda={lambd} is:", l2_penalty(model.w).item())
    except:
        print(f"L2 norm of w for lambda={lambd} is:", l2_penalty(model.net.weight).item())


num_inputs = 200 
for lambd in [0, 3]:
    model = WeightDecayFromScrath(num_inputs=num_inputs)
    train(lambd=lambd, model=model,
        optimizer = torch.optim.SGD(lr=0.01, params=model.params()),
        criterion = get_criterion(lambd),
    )

class ConciseLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, x): return self.net(x)

for lambd in [0, 3]:
    model = ConciseLinearModel()
    train(lambd=lambd, model=model,
        optimizer = torch.optim.SGD(lr=0.01, params=[
          {"params": model.net.weight, "weight_decay": lambd},
          {"params": model.net.bias}]),
        criterion = nn.MSELoss(),
    )
