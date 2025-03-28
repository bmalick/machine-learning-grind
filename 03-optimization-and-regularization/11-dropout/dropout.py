import torch
import torch.nn as nn


def dropout_layer(X, dropout, device=torch.device("cpu")):
    assert 0 <= dropout <= 1
    if dropout==1: return torch.zeros_like(X).to(device)
    mask = (torch.rand(X.shape).to(device)
            > dropout).float() # Uniform sampling
    return mask * X / (1.0 - dropout)

class DropoutMLPScratch(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_hiddens1: int, num_hiddens2: int,
                 dropout1: float, dropout2: float):
        super(DropoutMLPScratch, self).__init__()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.fc1 = nn.Linear(in_features=input_size, out_features=num_hiddens1)
        self.fc2 = nn.Linear(in_features=num_hiddens1, out_features=num_hiddens2)
        self.fc3 = nn.Linear(in_features=num_hiddens2, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x.reshape((x.shape[0], -1))))
        if self.training:
            h1 = dropout_layer(h1, self.dropout1, device=h1.device)
        h2 = self.relu(self.fc2(h1))
        if self.training:
            h2 = dropout_layer(h2, self.dropout2, device=h2.device)
        return self.fc3(h2)

class DropoutMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_hiddens1: int, num_hiddens2: int,
                 dropout1: float, dropout2: float):
        super(DropoutMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_hiddens1), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(in_features=num_hiddens1, out_features=num_hiddens2), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(in_features=num_hiddens2, out_features=output_size)
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))

