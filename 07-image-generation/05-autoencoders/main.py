import os
import glob
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
from argparse import ArgumentParser
import torch.nn.functional as F
import json

def show_images(batch: tuple[torch.Tensor]|list[torch.Tensor], nrow: int = 2, figsize: tuple[float, float] = (10.,8.), show: bool = True, save_name: str = None):
    if isinstance(batch, (tuple, list)):
        batch_imgs = batch[0]
    elif isinstance(batch, torch.Tensor):
        batch_imgs = batch
    else:
        print("ERROR SHOW IMAGES")
        return

    imgs = torchvision.utils.make_grid(batch_imgs, nrow=nrow)
    fig, ax = plt.subplots(figsize=figsize)
    # if isinstance(batch[0])
    ax.imshow(T.ToPILImage()(imgs))
    ax.axis("off")
    if save_name: fig.savefig(save_name, bbox_inches="tight", pad_inches=1)
    if show: plt.show()
    plt.close()
    
def get_mnist(batch_size: int = 8) -> torch.utils.data.DataLoader:
    mnist_data = torchvision.datasets.MNIST(
        root="mnist", train=True,
        download=True, transform=T.Compose([T.ToTensor()])
    )

    return torch.utils.data.DataLoader(
        mnist_data, batch_size=batch_size,
        shuffle=True
    )

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=4, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=4, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, kernel_size=5, stride=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(run_name: str,
    model: nn.Module, dataloader: torch.utils.data.DataLoader,
    num_epochs: int, lr: float,
    fixed_batch: torch.Tensor,
    device
):

    print("="*50)
    print(f"[Run {run_name}]")
    os.makedirs(run_name, exist_ok=True)

    model = model.to(device)
    criterion = lambda x,y: 0.5 * (x-y).pow(2).sum() / x.size(0)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def fit_epoch():
        model.train()
        num_instances = 0
        epoch_loss = 0.
        losses = []
        for step, batch in enumerate(dataloader):
            batch = [a.to(device) for a in batch]
            bs = batch[0].shape[0]
            num_instances += bs
            out = model(*batch[:-1])
            optimizer.zero_grad()
            # loss = criterion(out.flatten(), batch[0].flatten())
            loss = criterion(out, batch[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bs
            losses.append(loss.item())
        epoch_loss /= num_instances
        return epoch_loss, losses

    train_losses = []
    for epoch in range(num_epochs):
        train_loss, epoch_train_losses = fit_epoch()
        train_losses.extend(epoch_train_losses)
        print(f"[{epoch}/{num_epochs}] loss: {train_loss:.6f}")

        model.eval()
        with torch.no_grad():
            reconstructed = model(fixed_batch.to(device)).cpu()
            reconstructed = reconstructed.flatten(1)
            reconstructed = (reconstructed - reconstructed.min(1, keepdim=True).values) / (reconstructed.max(1, keepdim=True).values - reconstructed.min(1, keepdim=True).values + 1e-8)
            reconstructed = reconstructed.view_as(fixed_batch)
            show_images(reconstructed, nrow=12, figsize=(19.2,10.8), show=True, save_name=f"{run_name}/reconstructed-{epoch:03d}.jpg")

    frames = [Image.open(im) for im in sorted(glob.glob(f"{run_name}/reconstructed-*.jpg"))]
    frame_one = frames[0]
    frame_one.save(f"{run_name}/visu.gif", format="GIF", append_images=frames, save_all=True, duration=300, loop=0)

    plt.plot(train_losses)
    plt.savefig(f"{run_name}/train-loss.jpg")
    plt.close()

    with open(f"{run_name}/loss.json", "w") as f:
        json.dump({"train_loss": train_losses}, f)
    print("="*50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--latent-dim", "-d", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1.e-3)
    parser.add_argument("--num-epochs", type=int, default=25)
    args = parser.parse_args()

    lr = args.lr
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    num_epochs = args.num_epochs
    device = torch.device(args.device)
    run_name = f"latent-dim-{latent_dim}--hidden-dim-{hidden_dim}--lr-{lr}--epochs-{num_epochs}"

    os.makedirs(run_name, exist_ok=True)
    fixed_batch = next(iter(get_mnist(batch_size=36)))[0]
    show_images(fixed_batch, nrow=12, save_name=f"{run_name}/original-samples.jpg", figsize=(19.2,10.8), show=False)
    dataloader = get_mnist(batch_size=1024)

    model = AutoEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    train_autoencoder(run_name=run_name, model=model, dataloader=dataloader, num_epochs=num_epochs, lr=lr, fixed_batch=fixed_batch, device=device)

    fnames = glob.glob("latent-dim-*/*.json")
    names = [int(n.split("--")[0].split("-")[-1]) for n in fnames]
    metrics = {n: json.load(open(fn, "r")) for n, fn in zip(names, fnames)}
    metrics = dict(sorted(metrics.items(), key=lambda x:x[0]))
    for k, v in metrics.items():
        plt.plot(v["train_loss"], label=f"latent-dim = {k}")
    plt.legend()
    plt.title("train loss")
    plt.savefig("comparison.jpg")
    plt.close()
