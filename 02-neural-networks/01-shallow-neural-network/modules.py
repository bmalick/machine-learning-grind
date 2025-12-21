import torch
import numpy as np
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

## Hidden units regions

class HiddenUnit:
    def __init__(self, theta0, theta1, activation):
        self.theta0 = theta0
        self.theta1 = theta1
        self.activation = activation

    def __call__(self, x): return self.activation(self.theta0 + self.theta1 * x)
    
    def __str__(self): return "Hidden Unit: (%4.1f, %4.1f)" % (self.theta0, self.theta1)

class Relu:
    def __call__(self, x): return x if x>0 else 0
    

class ShallowNeuralNetwork:
    """1D shallow neural network"""
    def __init__(self, params, hidden_units_params):
        assert len(hidden_units_params) == len(params) - 1
        self.params = params
        self.hidden_units = [HiddenUnit(theta0=t0, theta1=t1, activation=Relu()) for t0, t1 in hidden_units_params]

    def __call__(self, x):
        res = self.params[0]
        for p,h in zip(self.params[1:], self.hidden_units):
            res += p * h(x)
        return res


def hidden_unit_linear_regions(hidden_unit, phi, axes, titles, color):
    ax1, ax2, ax3 = axes
    linear_func  = lambda x: hidden_unit.theta0 + hidden_unit.theta1 * x

    for ax in axes:
        ax.plot([0,2], [0,0], color="#abb2b9", linestyle="--")

    ax1.plot([0, 2], [linear_func(0), linear_func(2)], color=color)
    ax1.text(1., -0.85, titles[0], horizontalalignment='center',
             verticalalignment='center', color=color)
    ax2.text(1., -0.85, titles[1], horizontalalignment='center',
             verticalalignment='center', color=color)
    ax3.text(1., -0.85, titles[2], horizontalalignment='center',
             verticalalignment='center', color=color)

    t = np.linspace(0,2,50)
    ax2.plot(t, [hidden_unit(x) for x in t], color=color)
    ax3.plot(t, [phi*hidden_unit(x) for x in t], color=color)


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, w, b, n, noise=0.01):
        self.w = w
        self.b = b
        noise = noise * torch.randn(n, 1)
        self.X = torch.randn(n, len(w))
        self.y = torch.matmul(self.X, w.reshape(-1,1)) + b + noise
    
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def __len__(self): return len(self.X)


class FromScratchLinearModel:
    def __init__(self, num_inputs):
        self.w = torch.normal(mean=0., std=0.01, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def __call__(self, x): return self.forward(x)

class ConciseLinearModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, init=True):
        super().__init__()
        self.net = nn.Linear(in_features=input_size, out_features=output_size)
        if init:
            self.net.bias.data.fill_(0)
            self.net.weight.data.normal_(mean=0, std=1)
    def forward(self, x): return self.net(x)


class FashionMnist:
    def __init__(self, root="datasets", batch_size=16, resize=(28,28)):
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        train = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=trans
        )
        self.train = torch.utils.data.DataLoader(
            dataset=train, shuffle=True, batch_size=batch_size, num_workers=4
        )

def accuracy(y_pred, y_true, averaged=True):
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    preds = y_pred.argmax(axis=-1).type(y_true.dtype)
    compare = (preds==y_true.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition

class SoftmaxRegressionScratch:
    def __init__(self, num_inputs, num_outputs):
        self.w = torch.normal(mean=0, std=0.01, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
    
    def forward(self, *x):
        if isinstance(x, tuple): x = x[0]
        x = x.reshape((-1, self.w.shape[0]))
        return softmax(torch.matmul(x, self.w) + self.b)

    def __call__(self, x): return self.forward(x)

    def parameters(self): return [self.w, self.b]


class ConciseSoftmaxRegression(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_inputs, out_features=num_outputs)
        )
    def forward(self, x):
        return self.net(x)
    


# MLPs

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

class MLPScratch:
    def __init__(self, num_inputs: int, num_outputs: int, num_hiddens: int):
        self.W1 = nn.Parameter(data=torch.randn(size=(num_inputs, num_hiddens)) * 0.01)
        self.b1 = nn.Parameter(data=torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(data=torch.randn(size=(num_hiddens, num_outputs)) * 0.01, requires_grad=True)
        self.b2 = nn.Parameter(data=torch.zeros(num_outputs), requires_grad=True)
    
    def forward(self, x):
        if isinstance(x, tuple): x = x[0]
        x = x.reshape((-1, self.W1.shape[0]))
        H = relu(torch.matmul(x, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2
    
    def __call__(self, x): return self.forward(x)

    def parameters(self): return [self.W1, self.b1, self.W2, self.b2]


class ConciseMLP(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_hiddens: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_inputs, out_features=num_hiddens), nn.ReLU(),
            nn.Linear(in_features=num_hiddens, out_features=num_outputs),
        )

    def forward(self, x):
        return self.net(x)