import torch
import matplotlib.pyplot as plt


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



def train_model(model, criterion, data, epochs, optimizer, metrics=None):
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


def train2(model, criterion, data, epochs, optimizer, metrics=None):
    track_loss = []
    if metrics is not None:
        track_metrics = {m.__name__: [] for m in metrics}
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
            optimizer.step()
            epoch_loss += loss.item()
            if metrics is not None:
                for m in metrics:
                    epoch_metrics[m.__name__] += m(y_pred=y_pred, y_true=y_true)

        epoch_loss /= len(data)
        track_loss.append(epoch_loss)
        if metrics is not None:
            for m in metrics:
                epoch_metrics[m.__name__] /= len(data)
                track_metrics[m.__name__].append(epoch_metrics[m.__name__])
        summary = "Epoch: %d, loss: %.4f" % (i+1, epoch_loss)
        if metrics is not None:
            summary += ", " + ", ".join(["%s: %.4f" % (k,v) for k,v in epoch_metrics.items()])
        print(summary)
    plt.plot(track_loss); plt.title("loss"); plt.show()
    if metrics is not None:
        for name, values in track_metrics.items():
            plt.plot(values); plt.title(name); plt.show()
