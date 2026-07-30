import os
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

from data import show_images
from model import ddpm_inference

@dataclass
class TrainConfig:
    run_name: str = "ddpm"
    max_epochs: int = 100
    lr: float = 2e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    init_from: str = "scratch"
    resume_logdir: str|None = None
    run_id: str|None = None
    logdir: str|None = None
    save_checkpoint: bool = True
    save_model: bool = True
    model_save_fname: str|None = None

    figsize: tuple[float, float] = (8., 4.5)
    plot_interval: int = 5
    inference_interval: int = 1
    genviz: bool = True
    show_gen: bool = False

    def __post_init__(self):
        if self.init_from == "scratch":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{self.run_name}--{timestamp}"
            self.logdir = os.path.join("logs", self.run_id)
            os.makedirs("logs", exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)
            self.model_save_fname = os.path.join(self.logdir, self.run_name+".pt")

class Trainer:
    def __init__(self, config, datamodule, model, xT, init_train_info: dict|None = None):
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.init_train_info = init_train_info
        self.writer = SummaryWriter(log_dir=config.logdir)
        self.xT = xT.to(config.device)
        self.device = torch.device(config.device)

    def to_device(self, batch):
        return [a.to(self.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr)
        if self.init_train_info is not None:
            self.optimizer.load_state_dict(self.init_train_info["opt_state_dict"])

    def configure_metrics(self):
        self.metric_names = []
        self.metric_funcs = []

    @torch.no_grad
    def compute_metrics(self, y_hat, y_true):
        assert len(self.metric_names)==len(self.metric_funcs)
        out = {}
        if len(self.metric_names)==0: return {}
        for n, func in zip(self.metric_names, self.metric_funcs):
            out[n] = func(y_hat, y_true).item()
        return out

    def fit(self):
        self.model = self.model.to(self.device)
        self.configure_optimizers()
        self.configure_metrics()

        self.history = {
                **{n: {**{ln: {"train": [], "eval": []} for ln in ["loss", "elbo"]},
                   **{m: {"train": [], "eval": []} for m in self.metric_names}}
               for n in ["perstep", "perepoch"]}
        } if self.init_train_info is None else self.init_train_info["history"]

        epoch_num = 0 if self.init_train_info is None else self.init_train_info["epoch_num"] + 1
        self.best_eval_loss = 1e9 if self.init_train_info is None else self.init_train_info["best_eval_loss"]

        while True:
            self.train_step(epoch_num)
            self.eval_step(epoch_num)

            epoch_num += 1
            if epoch_num >= self.config.max_epochs:
                break

            if epoch_num > 0 and epoch_num % self.config.plot_interval == 0:
                for n in ["perstep", "perepoch"]:
                    self.make_plots(n)

        self.writer.close()
        self.save_model()
        self.save_history()
        for n in ["perstep", "perepoch"]:
            self.make_plots(n)

    def train_step(self, epoch_num):
        self.model.train()

        num_instances = 0
        epoch_history  = {k: 0.0 for k in self.history["perstep"]}

        for step_num, batch in enumerate(self.datamodule.train_dataloader):
            if isinstance(batch, (tuple, list)):
                x0,_ = self.to_device(batch)
            else:
                x0 = batch.to(self.device)
            eps0 = torch.randn(x0.size(), device=x0.device)
            timesteps = torch.randint(low=0, high=self.model.scheduler.T, size=(x0.size(0),1), device=x0.device)
            xt = self.model.scheduler.add_noise(x0, eps0, timesteps)
            out, loss = self.model(xt, timesteps, eps0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            B = batch[-1].size(0)
            num_instances += B

            epoch_history["loss"] += loss.item() * B
            epoch_history["elbo"] += - loss.item() * B

            self.history["perstep"]["loss"]["train"].append(loss.item())
            self.history["perstep"]["elbo"]["train"].append(-loss.item())

        for k, v in epoch_history.items():
            self.history["perepoch"][k]["train"].append(v / num_instances)

    def eval_step(self, epoch_num):
        self.model.eval()

        if self.config.genviz and (epoch_num % self.config.inference_interval == 0 or epoch_num==self.config.max_epochs-1):
            os.makedirs(os.path.join(self.config.logdir, "visualizations"), exist_ok=True)
            with torch.no_grad():
                reconstructed = ddpm_inference(self.model, self.xT)
                if self.config.run_name == "ddpm-simple":
                    reconstructed = reconstructed.detach().cpu().numpy()
                    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], s=5)
                    plt.savefig(os.path.join(self.config.logdir, f"visualizations/{epoch_num:03d}.jpg"))
                    plt.close()

                else:
                    show_images(reconstructed, nrow=12, figsize=(19.2,10.8), show=self.config.show_gen,
                                save_name=os.path.join(self.config.logdir, f"visualizations/{epoch_num:03d}.jpg"), white_bg=True)

        log_str = " | ".join([f"train_{k}: {v['train'][-1]:.5f}" for k, v in self.history["perepoch"].items()])
        print(f"Epoch: {epoch_num:3d} | {log_str}")

        if epoch_num > 0 and epoch_num % self.config.plot_interval == 0:
            checkpoint = {
                "epoch_num": epoch_num,
                "model": self.model.state_dict(),
            }
            log_str += f" | save checkpoint at {self.config.logdir}"
            torch.save(checkpoint, os.path.join(self.config.logdir, "ckpt.pt"))

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.state_dict(), self.config.model_save_fname)
            print(f"Model saved at {self.config.model_save_fname}")

    def save_history(self):
        fname = os.path.join(self.config.logdir, "history.json")
        with open(fname, "w") as f:
            json.dump(self.history, f)
        print(f"Save history at {fname}")

    def make_plots(self, plot_name: str):
        metrics = self.history[plot_name]
        if plot_name=="perepoch":
            for name, m_values in metrics.items():
                fig, ax = plt.subplots(figsize=self.config.figsize)
                for split,values in m_values.items():
                    if len(values) == 0: continue
                    ls = "-" if split=="train" else "--"
                    color = "blue" if split == "train" else "orange"
                    ax.plot(values, label=split, linestyle=ls, c=color)
                    ax.grid()
                ax.legend()
                ax.set_xlabel(plot_name.replace("per","")+"s")
                ax.set_title(name)
                fig.savefig(os.path.join(self.config.logdir, f"{name}-{plot_name}.jpg"))
                plt.close()
        elif plot_name=="perstep":
            for name, m_values in metrics.items():
                for split,values in m_values.items():
                    if len(values) == 0: continue
                    fig, ax = plt.subplots(figsize=self.config.figsize)
                    ls = "-" if split=="train" else "--"
                    color = "blue" if split == "train" else "orange"
                    ax.plot(values, label=split, linestyle=ls, c=color)
                    ax.grid()
                    ax.legend()
                    ax.set_title(name)
                    ax.set_xlabel(plot_name.replace("per","")+"s")
                    fig.savefig(os.path.join(self.config.logdir, f"{name}-{split}-{plot_name}.jpg"))
                    plt.close()

