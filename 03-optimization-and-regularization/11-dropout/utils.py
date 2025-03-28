import torch
import torchvision
import matplotlib.pyplot as plt


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout==1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float() # Uniform sampling
    return mask * X / (1.0 - dropout)


class MnistData:
    def __init__(self, root=".", train_batch_size=256, eval_batch_size=256, num_workers=4, resize=(28,28)):
        # self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])

        self.train = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=trans, download=True)
        self.eval = torchvision.datasets.FashionMNIST(
            root=root, train=False, transform=trans, download=True)

    def get_dataloader(self, train: bool):
        data = self.train if train else self.eval
        batch_size = self.train_batch_size if train else self.eval_batch_size
        return torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=train, num_workers=self.num_workers)

    def train_dataloader(self): return self.get_dataloader(train=True)
    def eval_dataloader(self): return self.get_dataloader(train=False)

def train(model, optimizer, criterion, data, epochs, device):
    print(f"Training on {device}")
    model = model.to(device)
    track_train_loss = []
    track_eval_loss = []
    track_train_acc = []
    track_eval_acc = []
    for e in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        num_instances = 0
        for batch in data.train_dataloader():
            batch = [x.to(device) for x in batch]
            output =model(*batch[:-1])
            loss = criterion(output, batch[-1])
            acc = (output.argmax(dim=-1)==batch[-1]).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch[-1].size(0)
            train_acc += acc.item() * batch[-1].size(0)
            num_instances += batch[-1].size(0)
        train_loss /= num_instances
        train_acc /= num_instances
        track_train_loss.append(train_loss)
        track_train_acc.append(train_acc)

        model.eval()
        eval_loss = 0
        eval_acc = 0
        num_instances = 0
        for batch in data.eval_dataloader():
            with torch.no_grad():
                batch = [x.to(device) for x in batch]
                output =model(*batch[:-1])
                loss = criterion(output, batch[-1])
                acc = (output.argmax(dim=-1)==batch[-1]).float().mean()
                eval_loss += loss.item() * batch[-1].size(0)
                eval_acc += acc.item() * batch[-1].size(0)
                num_instances += batch[-1].size(0)
        eval_loss /= num_instances
        eval_acc /= num_instances
        track_eval_loss.append(eval_loss)
        track_eval_acc.append(eval_acc)
        print(f"Epoch [{e+1}/{epochs}] loss: {train_loss:.5f}, acc: {train_acc:.5f}, eval_loss: {eval_loss:.5f}, eval_acc: {eval_acc:.5f}")
    return track_train_loss, track_eval_loss, track_train_acc, track_eval_acc

def plot_loss_and_metric(tracks):
    track_train_loss, track_eval_loss, track_train_acc, track_eval_acc = tracks
    fig, axes = plt.subplots(1,2, figsize=(10,4.5))
    axes[0].plot(track_train_loss, label="train")
    axes[0].plot(track_eval_loss, label="eval")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(track_train_acc, label="train")
    axes[1].plot(track_eval_acc, label="eval")
    axes[1].set_title("acc")
    axes[1].legend()
    plt.show(block=False)
    plt.pause(5)
    plt.close("all")
