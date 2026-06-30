import os
import math
import json
import torch
import torchvision
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


# torch.manual_seed(42)

# ----------------- Data -----------------

def show_images(batch: tuple[torch.Tensor]|list[torch.Tensor], nrow: int = 2, figsize: tuple[float, float] = (10.,8.), show: bool = True, save_name: str = None):
    if isinstance(batch, (tuple, list)):
        batch_imgs = batch[0]
    elif isinstance(batch, torch.Tensor):
        batch_imgs = batch
    else:
        print("ERROR SHOW IMAGES")
        return

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

# ----------------- Model -----------------

@dataclass
class ModuleConfig:
    latent_dim: int = 32
    hidden_dim: int = 32

class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.config = config

    def encode(self, x): raise NotImplementedError
    def decode(self, z): raise NotImplementedError

    def log_p_guassian(self, x, p_x_given_z):
        mu_ast, log_var_ast = p_x_given_z
        var_ast = log_var_ast.exp()
        return -0.5 * ( math.log(2 * math.pi) + log_var_ast + (x.view(x.size(0), -1)-mu_ast).pow(2) / var_ast ).sum(1)

    def log_p_bernoulli(self, x, p_x_given_z):
        p = p_x_given_z[0]
        p = p.view(p.size(0),-1)
        x = x.view(x.size(0),-1)
        return  (x * p.log() + (1-x) * (1-p).log()).sum(1)


    def kl_div_gaussian(self, q_z_given_x, prior):
        mu, log_var = q_z_given_x[0], q_z_given_x[1]
        mu_prior, log_var_prior = prior[0], prior[1]
        var = log_var.exp()
        var_prior = log_var_prior.exp()
        return 0.5 * (log_var_prior - log_var - 1 + (mu_prior.view(mu_prior.size(0), -1) - mu.view(mu.size(0), -1)).pow(2) / var_prior + var / var_prior).sum(1)


    def compute_loss(self, x, prior, p_x_given_z, q_z_given_x):
        log_p = self.log_p_guassian(x, p_x_given_z)
        kl_div = self.kl_div_gaussian(q_z_given_x, prior)
        loss = - (log_p - kl_div).mean()
        return loss, log_p.mean(), kl_div.mean()

    def forward(self, x, prior=None):
        q_z_given_x = self.encode(x)
        z = sample(q_z_given_x)
        p_x_given_z = self.decode(z)
        loss, log_p, kl_div = None, None, None
        if prior is not None:
            loss, log_p, kl_div = self.compute_loss(x, prior, p_x_given_z, q_z_given_x)
        return p_x_given_z, (loss, log_p, kl_div)

class VariationalAutoEncoderMPL(VariationalAutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=784, out_features=hidden_dim), nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=2*latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim), nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=784),
            nn.Sigmoid()
        )

    def encode(self, x):
        out = self.encoder(x.flatten(1)).view(x.size(0), 2, -1)
        mu, log_var = out[:, 0], out[:, 1]
        return mu, log_var

    def decode(self, z):
        mu = self.decoder(z)
        log_var = torch.zeros(mu.size(), device=mu.device)
        return mu, log_var


class VariationalAutoEncoderConv(VariationalAutoEncoder):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=2 * latent_dim, kernel_size=4, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=4, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, kernel_size=5, stride=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # [Key note] I used view here but conv output layers are not contiguous
        out = self.encoder(x).reshape(x.size(0), 2, -1)
        mu, log_var = out[:, 0], out[:, 1]
        return mu, log_var

    def decode(self, z):
        mu = self.decoder(z.reshape(z.size(0), -1, 1, 1))
        log_var = torch.zeros(mu.size(), device=mu.device)
        return mu, log_var

    def log_p_guassian(self, x, p_x_given_z):
        return super().log_p_guassian(x.flatten(1), (p_x_given_z[0].flatten(1), p_x_given_z[1].flatten(1)))

    def kl_div_gaussian(self, q_z_given_x, prior):
        return super().kl_div_gaussian(
                (q_z_given_x[0].flatten(1), q_z_given_x[1].flatten(1)),
                (prior[0].flatten(1), prior[1].flatten(1))
            )

# ----------------- Train -----------------

def gaussian_prior(latent_dim, device):
    mu = torch.zeros((1, latent_dim), device=device)
    log_var = torch.zeros((1, latent_dim), device=device)
    return (mu, log_var)

def sample(param):
    mu, log_var = param
    std = log_var.mul(0.5).exp()
    eps = torch.randn(mu.size(), device=mu.device)
    return mu + std * eps

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
    show_gen: bool = False
    latent_dim: int = 32

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
        assert self.config.latent_dim == self.model.config.latent_dim
        self.prior = gaussian_prior(latent_dim=config.latent_dim, device=config.device)

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

        self.losses = {
            "perstep": {"elbo": {"train": [], "eval": []}, "log_p": {"train": [], "eval": []}, "kl_div": {"train": [], "eval": []}},
            "perepoch": {"elbo": {"train": [], "eval": []}, "log_p": {"train": [], "eval": []}, "kl_div": {"train": [], "eval": []}},
        }

        for _ in range(self.config.max_epochs):
            self.train_step()
            self.eval_step()

            if self.current_epoch % self.config.eval_interval == 0:
                print(
                    f"Epoch: {self.current_epoch:3d} | "
                    f"train_elbo: {self.losses['perepoch']['elbo']['train'][-1]:.5f} | "
                    f"train_log_p: {self.losses['perepoch']['log_p']['train'][-1]:.5f} | "
                    f"train_kl_div: {self.losses['perepoch']['kl_div']['train'][-1]:.5f}"
                )

            self.current_epoch += 1

        self.writer.close()
        self.save_model()
        self.make_plots(self.losses)
        self.save_logs()

    def train_step(self):
        self.model.train()
        
        epoch_loss = 0.
        epoch_log_p = 0.
        epoch_kl_div = 0.

        num_instances = 0

        for batch in self.datamodule.train_dataloader:
            batch = self.to_device(batch)
            x = batch[0]
            p_x_given_z, (loss, log_p, kl_div) = self.model(x, self.prior)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs
            epoch_log_p += log_p.item() * bs
            epoch_kl_div += kl_div.item() * bs

            self.losses["perstep"]["elbo"]["train"].append(-loss.item())
            self.losses["perstep"]["log_p"]["train"].append(log_p.item())
            self.losses["perstep"]["kl_div"]["train"].append(kl_div.item())
            self.writer.add_scalar("perstep_elbo/train", -loss.item(), self.train_steps)
            self.writer.add_scalar("perstep_log_p/train", log_p.item(), self.train_steps)
            self.writer.add_scalar("perstep_kl_div/train", kl_div.item(), self.train_steps)


            self.train_steps += 1
        
        epoch_loss /= num_instances
        epoch_log_p /= num_instances
        epoch_kl_div /= num_instances

        self.losses["perepoch"]["elbo"]["train"].append(-epoch_loss)
        self.losses["perepoch"]["log_p"]["train"].append(epoch_log_p)
        self.losses["perepoch"]["kl_div"]["train"].append(epoch_kl_div)
        self.writer.add_scalar("perepoch_elbo/train", -epoch_loss, self.current_epoch)
        self.writer.add_scalar("perepoch_log_p/train", epoch_log_p, self.current_epoch)
        self.writer.add_scalar("perepoch_kl_div/train", epoch_kl_div, self.current_epoch)


    def eval_step(self):
        self.model.eval()
        if self.config.genviz:
            os.makedirs(os.path.join(self.config.logdir, "visualizations"), exist_ok=True)
            with torch.no_grad():
                p_x_given_z, _ = self.model(self.fixed_batch)
                reconstructed = p_x_given_z[0] # reparam trick -> var = 0
                reconstructed = reconstructed.view_as(self.fixed_batch)
                show_images(reconstructed, nrow=12, figsize=(19.2,10.8), show=self.config.show_gen,
                            save_name=os.path.join(self.config.logdir, f"visualizations/{self.current_epoch:03d}.jpg"))

    def make_plots(self, metrics):
        for plot_name, m_values in metrics.items():
            for name,values in m_values.items():
                fig, ax = plt.subplots(figsize=self.config.figsize)
                for split, v in values.items():
                    if self.config.figlog:
                        ax.semilogy(v, label=split, linestyle="-" if split=="train" else "--")
                    else:
                        ax.plot(v, label=split, linestyle="-" if split=="train" else "--")
                    if self.config.figgrid: ax.grid()
                ax.legend()
                ax.set_xlabel(plot_name.replace("per","")+"s")
                ax.set_title(name)
                fig.savefig(os.path.join(self.config.logdir, f"{name}-{plot_name}.jpg"))
                plt.close()
            plt.close()

    def compare_plots(self, metrics_a, metrics_b, other_trainer):
        for pername in ["perstep", "perepoch"]:
            for loss_name in metrics_a[pername]:
                fig, ax = plt.subplots(figsize=self.config.figsize)
                for (split1,value1), (split2,value2) in zip(metrics_a[pername][loss_name].items(), metrics_b[pername][loss_name].items()):
                    if self.config.figlog:
                        if len(value1):
                            ax.semilogy(value1, label=f"{self.config.run_name} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                        if len(value2):
                            ax.semilogy(value2, label=f"{other_trainer.config.run_name} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
                    else:
                        if len(value1):
                            ax.plot(value1, label=f"{self.config.run_name} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                        if len(value2):
                            ax.plot(value2, label=f"{other_trainer.config.run_name} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
                    if self.config.figgrid: ax.grid()
                ax.legend()
                ax.set_xlabel(pername.replace("per","")+"s")
                ax.set_title(loss_name)
                compare_dir = f"{self.config.run_id}-vs-{other_trainer.config.run_id}"
                os.makedirs(compare_dir, exist_ok=True)
                fig.savefig(os.path.join(compare_dir, f"{pername}-{loss_name}.jpg"))
                plt.close()

    def save_logs(self):
        fname = os.path.join(self.config.logdir, "losses.json")
        with open(fname, "w") as f:
            json.dump(self.losses, f)
        print(f"Save losses at {fname}")

if __name__ == "__main__":
    # !rm -rf logs

    datamodule = DataModule(DataConfig())
    auto_enc_mlp = VariationalAutoEncoderMPL(ModuleConfig(hidden_dim=500))
    mlp_trainer = Trainer(TrainConfig(run_name="mlp_auto_enc"), datamodule, auto_enc_mlp)
    mlp_trainer.fit()

    auto_enc_conv = VariationalAutoEncoderConv(ModuleConfig(hidden_dim=32))
    conv_trainer = Trainer(TrainConfig(run_name="conv_auto_enc"), datamodule, auto_enc_conv)
    conv_trainer.fit()

    mlp_trainer.compare_plots(mlp_trainer.losses, conv_trainer.losses, conv_trainer)

