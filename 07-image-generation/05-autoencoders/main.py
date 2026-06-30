import os
import math
import json
import torch
import torchvision
from torch import nn
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)

# ----------------- Data -----------------

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

@dataclass
class DataConfig:
    train_batch_size: int = 1024
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

# ----------------- Model -----------------

@dataclass
class ModuleConfig:
    latent_dim: int = 2
    hidden_dim: int = 32


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
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

    def compute_loss(self, y_hat, y_true):
        return 0.5 * (y_hat-y_true).pow(2).sum() / y_hat.size(0)

    def forward(self, x, targets=None):
        out = self.decoder(self.encoder(x))
        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss


# ----------------- Train -----------------

@dataclass
class TrainConfig:
    run_name: str = "auto_enc"
    max_epochs: int = 25
    eval_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-3
    save_model: bool = True
    figsize: tuple[float, float] = (8., 4.5)
    figlog: bool = False
    figgrid: bool = True
    genviz: bool = True

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{self.run_name}--{timestamp}"
        self.logdir = os.path.join("logs", self.run_id)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        self.model_save_fname = os.path.join(self.logdir, self.run_name+".pth")

class Trainer:
    def __init__(self, config, datamodule, model):
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.writer = SummaryWriter(log_dir=config.logdir)
        self.fixed_batch = next(iter(DataModule(DataConfig(train_batch_size=36)).train_dataloader))[0].to(config.device)

    def to_device(self, batch):
        return [a.to(self.config.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.state_dict(), self.config.model_save_fname)
            print(f"Model saved ad {self.config.model_save_fname}")

    def fit(self):
        self.model = self.model.to(self.config.device)
        self.configure_optimizers()

        self.current_epoch = 0
        self.train_steps = 0
        self.eval_steps = 0

        self.perstep_losses = {"train": [], "eval": []}
        self.perepoch_losses = {"train": [], "eval": []}

        for _ in range(self.config.max_epochs):
            self.train_step()
            self.eval_step()

            if self.current_epoch % self.config.eval_interval == 0:
                print(
                    f"Epoch: {self.current_epoch:3d} | "
                    f"train_loss: {self.perepoch_losses['train'][-1]:.5f} | "
                )

            self.current_epoch += 1

        self.writer.close()
        self.save_model()
        self.make_plots()
        self.save_logs()

    def train_step(self):
        self.model.train()
        
        epoch_loss = 0.
        num_instances = 0

        for batch in self.datamodule.train_dataloader:
            batch = self.to_device(batch)
            x = batch[0]
            out, loss = self.model(x, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs

            self.perstep_losses["train"].append(loss.item())
            self.writer.add_scalar("perstep_loss/train", loss.item(), self.train_steps)


            self.train_steps += 1
        
        epoch_loss /= num_instances
        self.perepoch_losses["train"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/train", epoch_loss, self.current_epoch)


    def eval_step(self):
        self.model.eval()
        if self.config.genviz:
            os.makedirs(os.path.join(self.config.logdir, "visualizations"), exist_ok=True)
            with torch.no_grad():
                reconstructed, _ = self.model(self.fixed_batch)
                reconstructed = reconstructed.cpu().flatten(1)
                reconstructed = (reconstructed - reconstructed.min(1, keepdim=True).values) / (reconstructed.max(1, keepdim=True).values - reconstructed.min(1, keepdim=True).values + 1e-8)
                reconstructed = reconstructed.view_as(self.fixed_batch)
                show_images(reconstructed, nrow=12, figsize=(19.2,10.8), show=True,
                            save_name=os.path.join(self.config.logdir, f"visualizations/{self.current_epoch:03d}.jpg"))

        # frames = [Image.open(im) for im in sorted(glob.glob(f"{run_name}/reconstructed-*.jpg"))]
        # frame_one = frames[0]
        # frame_one.save(f"{run_name}/visu.gif", format="GIF", append_images=frames, save_all=True, duration=300, loop=0)


    def make_plots(self):
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for split, values in self.perepoch_losses.items():
            if self.config.figlog:
                ax.semilogy(values, label=split, linestyle="-" if split=="train" else "--")
            else:
                ax.plot(values, label=split, linestyle="-" if split=="train" else "--")
            if self.config.figgrid: ax.grid()
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_title("loss")
        fig.savefig(os.path.join(self.config.logdir, "loss.jpg"))
        plt.close()


    def save_logs(self):
        fname = os.path.join(self.config.logdir, "losses.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_losses, "perepoch": self.perepoch_losses}, f)
        print(f"Save losses at {fname}")

if __name__ == "__main__":
    datamodule = DataModule(DataConfig())
    auto_enc = AutoEncoder(ModuleConfig())
    trainer = Trainer(TrainConfig(), datamodule, auto_enc)
    trainer.fit()
