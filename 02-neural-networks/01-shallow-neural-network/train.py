import torch

class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, w, b, n, noise=0.01):
        self.w = w
        self.b = b
        noise = noise * torch.randn(n, 1)
        self.X = torch.randn(n, len(w))
        self.y = torch.matmul(self.X, w.reshape(-1,1)) + b + noise
    
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def __len__(self): return len(self.X)


class LinearModel:
    def __init__(self, num_inputs):
        self.w = torch.normal(mean=0., std=0.01, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def __call__(self, x): return self.forward(x)

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

def train(model, criterion, data, epochs, optimizer):
    for i in range(epochs):
        epoch_loss = 0
        for batch in data:
            loss = criterion(model(*batch[:-1]), batch[-1])
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        print("Epoch: %d, loss: %.4f" % (i+1, epoch_loss))


if __name__ == "__main__":
    data = LinearDataset(w=torch.tensor([1.3, 0.6]),
                         b=torch.tensor([4.9]), n=1000, noise=0.01)
    dataloader = torch.utils.data.DataLoader(dataset=data, shuffle=True, batch_size=16, num_workers=4)
    print("num of batches: ", len(dataloader))
    model = LinearModel(num_inputs=2)
    optimizer = SGD(params=[model.w, model.b], lr=0.01)
    criterion = lambda x,y: ((x-y)**2 / 2).mean()

    print("Params to estimate are:")
    print("w:", dataloader.dataset.w, ", b:", dataloader.dataset.b)
    print("Before training:")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())
    train(model, criterion, dataloader, 10, optimizer)
    print("After training")
    print("w:", model.w.detach().numpy(), ", b:", model.b.item())
