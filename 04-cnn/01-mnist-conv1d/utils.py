import torch
import torchvision

class MnistData:
    def __init__(self, root=".", train_batch_size=128, eval_batch_size=128, num_workers=4, resize=(28,28)):
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


def compute_number_of_parameters(model):
    count = 0
    for _,w in model.named_parameters():
        count+=w.shape.numel()
    return count


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

