import torch
import torchvision
from torchvision import transforms

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

def cross_entropy(y_pred, y_true):
    return -torch.log(y_pred[list(range(len(y_true))), y_true]).mean()

def train(model, criterion, data, epochs, optimizer, metrics=None):
    for i in range(epochs):
        epoch_loss = 0
        if metrics is not None:
            epoch_metrics = {m.__name__: 0 for m in metrics}
        for batch in data:
            y_pred = model(*batch[:-1])
            y_true = batch[-1]
            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                optimizer.step()
            epoch_loss += loss.item()
            if metrics is not None:
                for m in metrics:
                    epoch_metrics[m.__name__] += m(y_pred=y_pred, y_true=y_true)

        epoch_loss /= len(data)
        if metrics is not None:
            for m in metrics:
                epoch_metrics[m.__name__] /= len(data)
        summary = "Epoch: %d, loss: %.4f" % (i+1, epoch_loss)
        if metrics is not None:
            summary += ", " + ", ".join(["%s: %.4f" % (k,v) for k,v in epoch_metrics.items()])
        print(summary)


if __name__ == "__main__":
    print("#### Linear regression")
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

    print("\n\n#### Linear classification")
    model = SoftmaxRegressionScratch(num_inputs=28*28, num_outputs=10)
    dataloader = FashionMnist(batch_size=256).train
    criterion = cross_entropy
    epochs=10
    optimizer = SGD(model.parameters(), 0.1)
    train(model=model, data=dataloader, criterion=criterion, optimizer=optimizer, epochs=10, metrics=[accuracy])
