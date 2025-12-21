import torch

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

