import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

class MnistData:
    def __init__(self, root: str = ".", batch_size: int = 128, resize: tuple = (28,28), num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor(),
        ])
        self.train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transforms, download=True)
        self.eval= torchvision.datasets.FashionMNIST(root=root, train=False, transform=transforms, download=True)

    def get_dataloader(self, train: bool):
        return torch.utils.data.DataLoader(
            dataset=self.train if train else self.eval,
            batch_size=self.batch_size, shuffle=train,
            num_workers=self.num_workers)

    def train_dataloader(self): return self.get_dataloader(train=True)
    def eval_dataloader(self): return self.get_dataloader(train=False)

def train(model, learning_rate, epochs, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lr=learning_rate, params=model.parameters())

    tracking = {
        "loss": {"train": [], "eval": []},
        "acc": {"train": [], "eval": []}
    }

    model = model.to(device)

    for epoch_num in range(epochs):
        train_loss = 0.
        eval_loss = 0.
        train_acc = 0.
        eval_acc = 0.
        train_instances = 0
        eval_instances = 0

        model.train()
        for batch in data.train_dataloader():
            batch = [b.to(device) for b in batch]
            output = model(*batch[:-1])
            loss = criterion(output, batch[-1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += batch[-1].size(0) * loss.item()
            train_instances += batch[-1].size(0)
            acc = (output.argmax(dim=-1) == batch[-1]).float().mean()
            train_acc += batch[-1].size(0) * acc.item()
            print(f"[{epoch_num+1}/{epochs}] loss: {train_loss/train_instances:.5f}, acc: {train_acc/train_instances:.5f}")

        model.eval()
        for batch in data.eval_dataloader():
            batch = [b.to(device) for b in batch]
            output = model(*batch[:-1])
            loss = criterion(output, batch[-1])
            eval_loss += batch[-1].size(0) * loss.item()
            eval_instances += batch[-1].size(0)
            acc = (output.argmax(dim=-1) == batch[-1]).float().mean()
            eval_acc += batch[-1].size(0) * acc.item()

        print(f"[End of {epoch_num+1}/{epochs}] eval_loss: {eval_loss/eval_instances:.5f}, eval_acc: {eval_acc/eval_instances:.5f}")

        tracking["loss"]["train"].append(train_loss / train_instances)
        tracking["acc"]["train"].append(train_acc / train_instances)
        tracking["loss"]["eval"].append(eval_loss / eval_instances)
        tracking["acc"]["eval"].append(eval_acc / eval_instances)

    fig, axes = plt.subplots(1, 2, figsize=(10,4.5))
    for n, x in tracking["loss"].items():
        axes[0].plot(x, label=n)
    for n, x in tracking["acc"].items():
        axes[1].plot(x, label=n)
    axes[0].set_title("loss")
    axes[1].set_title("acc")
    axes[0].legend()
    axes[1].legend()
    plt.title("LeNet")
    plt.savefig("loss-and-acc.png")
    plt.show(block=False)
    plt.pause(3)
    print("Figure saved in pwd")
    plt.close()




