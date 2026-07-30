import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torchvision.transforms as T


def show_images(batch: tuple[torch.Tensor]|list[torch.Tensor], nrow: int = 2, figsize: tuple[float, float] = (10.,8.), show: bool = True, save_name: str = None, white_bg: bool = False):
    if isinstance(batch, (tuple, list)):
        batch_imgs = batch[0]
    elif isinstance(batch, torch.Tensor):
        batch_imgs = batch
    else:
        print("ERROR SHOW IMAGES")
        return

    batch_imgs = (batch_imgs + 1.0) / 2.0
    batch_imgs = torch.clamp(batch_imgs, 0.0, 1.0)

    imgs = torchvision.utils.make_grid(batch_imgs, nrow=nrow)
    if white_bg:
        imgs = 1 - imgs
    fig, ax = plt.subplots(figsize=figsize)
    # if isinstance(batch[0])
    ax.imshow(T.ToPILImage()(imgs))
    ax.axis("off")
    if save_name: fig.savefig(save_name, bbox_inches="tight", pad_inches=1)
    if show: plt.show()
    plt.close()

@dataclass
class MnistConfig:
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_workers: int = 2
    root: str = "./mnist"

class Mnist:
    def __init__(self, config):
        self.config = config

        data = torchvision.datasets.MNIST(
            root=config.root, train=True,
            download=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        )

        self.train_dataloader = torch.utils.data.DataLoader(dataset=data,
                batch_size=self.config.train_batch_size, shuffle=True,
                num_workers=self.config.num_workers)

@dataclass
class SpiralDataConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 2
    n_samples: int = 1000
    noise: float = 0.01
    coef: float = 1.

class FermatSpiralData:
    def __init__(self, config):
        self.config = config
        n_samples = config.n_samples
        noise = config.noise

        theta = np.linspace(0, 5 * np.pi, n_samples)
        r = config.coef * np.sqrt(theta)
        x = r * np.cos(theta) + np.random.randn(n_samples) * noise
        y = r * np.sin(theta) + np.random.randn(n_samples) * noise
        p = np.stack([x, y], axis=1)

        p = (p - p.mean(axis=0)) / p.std(axis=0)
        data = torch.tensor(p, dtype=torch.float32)

        self.train_dataloader = torch.utils.data.DataLoader(dataset=data,
                batch_size=config.train_batch_size, shuffle=True,
                num_workers=config.num_workers)

if __name__ == "__main__":
    datamodule = FermatSpiralData(SpiralDataConfig(n_samples=1000))
    x = datamodule.train_dataloader.dataset.numpy()
    plt.scatter(x[:, 0], x[:, 1], s=5)
    plt.show()

