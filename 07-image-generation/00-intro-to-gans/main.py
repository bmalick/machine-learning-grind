import os
import json
import torch
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)

# ----------------- Data -----------------

@dataclass
class DataConfig:
    n_data: int = 1000
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2

class DataModule:
    def __init__(self, config):
        X = torch.normal(0.0, 1, (config.n_data, 2))
        A = torch.tensor([[1, 2], [-0.1, 0.5]])
        b = torch.tensor([1, 2])
        self.data = torch.matmul(X, A) + b

        self.config = config

        self.train_dataloader =  torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(self.data),
                batch_size=config.train_batch_size, shuffle=True)


# ----------------- Model -----------------

@dataclass
class ModuleConfig:
    latent_dim: int = 2
    input_dim: int = 2



class GAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.G = nn.Sequential(
            nn.Linear(config.latent_dim, config.input_dim)
        )

        self.D = nn.Sequential(
            nn.Linear(config.input_dim, 5), nn.Tanh(),
            nn.Linear(5, 3), nn.Tanh(),
            nn.Linear(3, 1)
        )


# ----------------- Train -----------------

def update_discriminator(x, z, D, G, criterion, trainer_D):
    """Update discriminator."""
    batch_size = x.shape[0]
    ones = torch.ones((batch_size,), device=x.device)
    zeros = torch.zeros((batch_size,), device=x.device)

    trainer_D.zero_grad()

    real_y = D(x)
    fake_x = G(z)
    fake_y = D(fake_x.detach())
    loss_D = (criterion(real_y, ones.reshape(real_y.shape)) +
                          criterion(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_generator(z, D, G, criterion, trainer_G):
    """Update generator."""
    batch_size = z.shape[0]
    ones = torch.ones((batch_size,), device=z.device)

    trainer_G.zero_grad()

    fake_x = G(z)
    fake_y = D(fake_x)
    loss_G = criterion(fake_y, ones.reshape(fake_y.shape))
    loss_G.backward()
    trainer_G.step()
    return  loss_G

@dataclass
class TrainConfig:
    run_name: str = "gan"
    latent_dim: int = 2
    max_epochs: int = 30
    eval_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr_D: float = 0.05
    lr_G: float = 0.005
    save_model: bool = True
    figsize: tuple[float, float] = (8., 4.5)
    figlog: bool = False
    figgrid: bool = True
    genviz: bool = True
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{self.run_name}--{timestamp}"
        self.logdir = os.path.join("logs", self.run_id)
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        self.model_G_save_fname = os.path.join(self.logdir, self.run_name+"_G.pth")
        self.model_D_save_fname = os.path.join(self.logdir, self.run_name+"_D.pth")

class Trainer:
    def __init__(self, config, datamodule, model):
        assert config.latent_dim==model.config.latent_dim
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.fixed_noise = torch.normal(0, 1, size=(100, config.latent_dim), device=config.device)

    def to_device(self, batch):
        return [a.to(self.config.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer_D = torch.optim.Adam(self.model.D.parameters(), self.config.lr_D)
        self.optimizer_G = torch.optim.Adam(self.model.G.parameters(), self.config.lr_G)

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.G.state_dict(), self.config.model_G_save_fname)
            torch.save(self.model.D.state_dict(), self.config.model_D_save_fname)
            print(f"Model generator saved ad {self.config.model_G_save_fname}")
            print(f"Model discriminator saved ad {self.config.model_D_save_fname}")

    def fit(self):
        self.model.D = self.model.D.to(self.config.device)
        self.model.G = self.model.G.to(self.config.device)
        self.configure_optimizers()

        self.current_epoch = 0
        self.train_steps = 0
        self.eval_steps = 0

        self.perstep_losses = {"loss_D": {"train": [], "eval": []}, "loss_G": {"train": [], "eval": []}}
        self.perepoch_losses = {"loss_D": {"train": [], "eval": []}, "loss_G": {"train": [], "eval": []}}

        for _ in range(self.config.max_epochs):
            self.train_step()

            if self.current_epoch % self.config.eval_interval == 0:
                print(
                    f"Epoch: {self.current_epoch:3d} | "
                    f"train_loss_D: {self.perepoch_losses['loss_D']['train'][-1]:.5f} | "
                    f"train_loss_G: {self.perepoch_losses['loss_G']['train'][-1]:.5f} | "
                )

            self.current_epoch += 1

        self.writer.close()
        self.save_model()
        self.make_plots()
        self.save_logs()

    def train_step(self):
        self.model.train()
        
        epoch_loss_D = 0.
        epoch_loss_G = 0.
        num_instances = 0

        for (X,) in self.datamodule.train_dataloader:
            (X,) = self.to_device((X,))
            bs = X.size(0)

            Z = torch.normal(0, 1, size=(bs, self.config.latent_dim), device=X.device)
            loss_D = update_discriminator(X, Z, self.model.D, self.model.G, self.config.criterion, self.optimizer_D).item()
            loss_G = update_generator(Z, self.model.D, self.model.G, self.config.criterion, self.optimizer_G).item()

            num_instances += bs
            epoch_loss_D += loss_D * bs
            epoch_loss_G += loss_G * bs

            self.perstep_losses["loss_D"]["train"].append(loss_D / num_instances)
            self.perstep_losses["loss_G"]["train"].append(loss_G / num_instances)
            self.writer.add_scalar("perstep_loss_D/train", loss_D / num_instances, self.train_steps)
            self.writer.add_scalar("perstep_loss_G/train", loss_G / num_instances, self.train_steps)


            self.train_steps += 1
        
        epoch_loss_D /= num_instances
        epoch_loss_G /= num_instances
        self.perepoch_losses["loss_D"]["train"].append(epoch_loss_D)
        self.perepoch_losses["loss_G"]["train"].append(epoch_loss_G)
        self.writer.add_scalar("perepoch_loss_D/train", epoch_loss_D, self.current_epoch)
        self.writer.add_scalar("perepoch_loss_G/train", epoch_loss_G, self.current_epoch)


        if self.config.genviz:
            os.makedirs(os.path.join(self.config.logdir, "visualizations"), exist_ok=True)
            fake_data = self.model.G(self.fixed_noise).detach().cpu().numpy()
            plt.scatter(self.datamodule.data[:100, 0], self.datamodule.data[:100, 1], label="real")
            plt.scatter(fake_data[:, 0], fake_data[:, 1], label="generated")
            plt.legend(["real", "generated"])
            plt.savefig(os.path.join(self.config.logdir, f"visualizations/{self.current_epoch:02d}.png"))
            plt.close()


    def make_plots(self):
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for name, values in self.perepoch_losses.items():
            if self.config.figlog:
                ax.semilogy(values["train"], label=name)
            else:
                ax.plot(values["train"], label=name)
            if self.config.figgrid: ax.grid()
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_title("loss")
        fig.savefig(os.path.join(self.config.logdir, "losses.jpg"))
        plt.close()


    def save_logs(self):
        fname = os.path.join(self.config.logdir, "losses.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_losses, "perepoch": self.perepoch_losses}, f)
        print(f"Save losses at {fname}")


if __name__ == "__main__":
    datamodule = DataModule(DataConfig())
    gan  = GAN(ModuleConfig(input_dim=datamodule.data.shape[1]))
    trainer = Trainer(TrainConfig(), datamodule, gan)
    trainer.fit()
