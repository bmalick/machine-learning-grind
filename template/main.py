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
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)

# ----------------- Data -----------------

@dataclass
class DataConfig:
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2

class DataModule:
    def __init__(self, config):
        self.config = config

        self.train = None
        self.eval = None
        self.train_dataloader = self.get_dataloader(True)
        self.eval_dataloader = self.get_dataloader(False)

    def get_dataloader(self, train: bool):
        return torch.utils.data.DataLoader(
                dataset=self.train if train else self.eval,
                batch_size=self.config.train_batch_size if train else self.config.eval_batch_size,
                shuffle=train,
                num_workers=self.config.num_workers)


# ----------------- Model -----------------

@dataclass
class ModuleConfig:
    num_blocks: int = 2


class Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def compute_loss(self, y_hat, y_true):
        raise NotImplementedError

    def number_of_params():
        pass

    def forward(self, x, targets=None):
        out = self.net(x)
        loss = None
        if targets is not None:
            loss = self.compute_loss(out, targets)
        return out, loss


# ----------------- Train -----------------

@dataclass
class TrainConfig:
    run_name: str = "run"
    max_epochs: int = 10
    eval_interval: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-3
    save_model: bool = True
    figsize: tuple[float, float] = (8., 4.5)
    figlog: bool = False
    figgrid: bool = True

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

    def to_device(self, batch):
        return [a.to(self.config.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.config.learning_rate)

    def configure_metrics(self):
        # self.metric_names = []
        # self.metric_funcs = [
        # ]
        self.perstep_metrics = {n: {"train": [], "eval": []} for n in self.metric_names}
        self.perepoch_metrics = {n: {"train": [], "eval": []} for n in self.metric_names}

    def compute_metrics(self, y_hat, y_true):
        assert len(self.metric_names)==len(self.metric_funcs)
        out = {}
        if len(self.metric_names)==0: return {}
        for n, func in zip(self.metric_names, self.metric_funcs):
            out[n] = func(y_hat, y_true).item()
        return out

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.state_dict(), self.config.model_save_fname)
            print(f"Model saved ad {self.config.model_save_fname}")

    def fit(self):
        self.model = self.model.to(self.config.device)
        self.configure_optimizers()
        self.configure_metrics()

        self.current_epoch = 0
        self.train_steps = 0
        self.eval_steps = 0

        self.perstep_losses = {"train": [], "eval": []}
        self.perepoch_losses = {"train": [], "eval": []}

        for _ in range(self.config.max_epochs):
            self.train_step()
            self.eval_step()

            if self.current_epoch % self.config.eval_interval == 0:
                metrics_str = " | ".join([f"train_{k}: {v['train'][-1]:.5f} | eval_{k}: {v['eval'][-1]:.5f}" for k, v in self.perepoch_metrics.items()])
                print(
                    f"Epoch: {self.current_epoch:3d} | "
                    f"train_loss: {self.perepoch_losses['train'][-1]:.5f} | "
                    f"eval_loss: {self.perepoch_losses['eval'][-1]:.5f} | "
                    f"{metrics_str}"
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
        epoch_metrics = {n: 0. for n in self.metric_names}

        for batch in self.datamodule.train_dataloader:
            batch = self.to_device(batch)
            out, loss = self.model(*batch[:-1], batch[-1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = batch[-1].size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs

            metrics = self.compute_metrics(out, batch[-1])
            for k,v in metrics.items():
                self.perstep_metrics[k]["train"].append(v)
                self.writer.add_scalar(f"perstep_{k}/train", v, self.train_steps)
                epoch_metrics[k] += v * bs

            self.perstep_losses["train"].append(loss.item())
            self.writer.add_scalar("perstep_loss/train", loss.item(), self.train_steps)


            self.train_steps += 1
        
        for k,v in epoch_metrics.items():
            self.perepoch_metrics[k]["train"].append(v/num_instances )
            self.writer.add_scalar(f"perepoch_{k}/train", v/num_instances, self.current_epoch)

        epoch_loss /= num_instances
        self.perepoch_losses["train"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/train", epoch_loss, self.current_epoch)


    def eval_step(self):
        self.model.eval()

        epoch_loss = 0.
        num_instances = 0
        epoch_metrics = {n: 0. for n in self.metric_names}

        for batch in self.datamodule.eval_dataloader:
            batch = self.to_device(batch)
            with torch.no_grad():
                out, loss = self.model(*batch[:-1], batch[-1])

            bs = batch[-1].size(0)
            num_instances += bs
            epoch_loss += loss.item() * bs

            metrics = self.compute_metrics(out, batch[-1])
            for k,v in metrics.items():
                self.perstep_metrics[k]["eval"].append(v)
                self.writer.add_scalar(f"perstep_{k}/eval", v, self.eval_steps)
                epoch_metrics[k] += v * bs

            self.perstep_losses["eval"].append(loss.item())
            self.writer.add_scalar("perstep_loss/eval", loss.item(), self.eval_steps)

            self.eval_steps += 1
        
        for k,v in epoch_metrics.items():
            self.perepoch_metrics[k]["eval"].append(v/num_instances )
            self.writer.add_scalar(f"perepoch_{k}/eval", v/num_instances, self.current_epoch)

        epoch_loss /= num_instances
        self.perepoch_losses["eval"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/eval", epoch_loss, self.current_epoch)

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

        for name,values in self.perepoch_metrics.items():
            fig, ax = plt.subplots(figsize=self.config.figsize)
            for split, v in values.items():
                if self.config.figlog:
                    ax.semilogy(v, label=split, linestyle="-" if split=="train" else "--")
                else:
                    ax.plot(v, label=split, linestyle="-" if split=="train" else "--")
                if self.config.figgrid: ax.grid()
            ax.legend()
            ax.set_xlabel("epochs")
            ax.set_title(name)
            fig.savefig(os.path.join(self.config.logdir, f"{name}.jpg"))
            plt.close()

    def save_logs(self):
        fname = os.path.join(self.config.logdir, "losses.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_losses, "perepoch": self.perepoch_losses}, f)
        print(f"Save losses at {fname}")

        fname = os.path.join(self.config.logdir, "metrics.json")
        with open(fname, "w") as f:
            json.dump({"perstep": self.perstep_metrics, "perepoch": self.perepoch_metrics}, f)
        print(f"Save metrics at {fname}")

if __name__ == "__main__":
    datamodule = DataModule(DataConfig())
    module = Module(ModuleConfig())
    trainer = Trainer(TrainConfig(), datamodule, module)
    trainer.fit()
