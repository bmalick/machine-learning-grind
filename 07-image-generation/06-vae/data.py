import torch
import torchvision
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torchvision.transforms as T


def show_images(batch: tuple[torch.Tensor]|list[torch.Tensor], nrow: int = 2, figsize: tuple[float, float] = (10.,8.), show: bool = True, save_name: str = None):
    if isinstance(batch, (tuple, list)):
        batch_imgs = batch[0]
    elif isinstance(batch, torch.Tensor):
        batch_imgs = batch
    else:
        print("ERROR SHOW IMAGES")
        return

    # imgs = torchvision.utils.make_grid(batch_imgs, nrow=nrow)
    imgs = 1-torchvision.utils.make_grid(batch_imgs, nrow=nrow)
    fig, ax = plt.subplots(figsize=figsize)
    # if isinstance(batch[0])
    ax.imshow(T.ToPILImage()(imgs))
    ax.axis("off")
    if save_name: fig.savefig(save_name, bbox_inches="tight", pad_inches=1)
    if show: plt.show()
    plt.close()

@dataclass
class DataConfig:
    train_batch_size: int = 100
    eval_batch_size: int = 8
    num_workers: int = 2
    root: str = "./mnist"

class DataModule:
    def __init__(self, config):
        self.config = config

        data = torchvision.datasets.MNIST(
            root=config.root, train=True,
            download=True, transform=T.Compose([T.ToTensor()])
        )

        self.train_dataloader = torch.utils.data.DataLoader(dataset=data,
                batch_size=self.config.train_batch_size, shuffle=True,
                num_workers=self.config.num_workers)
