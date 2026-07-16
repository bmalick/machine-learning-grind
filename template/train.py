import os
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
import dataclasses
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainConfig:
    run_name: str = "run"
    max_epochs: int = 100
    max_iters: int = 60000
    weight_decay: float = 1e-2
    lr: float = 1e-3
    weight_decay: float = 0.
    grad_accum_steps: int = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_from: str = "scratch"
    resume_logdir: str|None = None
    run_id: str|None = None
    logdir: str|None = None
    save_checkpoint: bool = True
    save_model: bool = True
    model_save_fname: str|None = None

    figsize: tuple[float, float] = (8., 4.5)
    plot_interval: int = 5

    def __post_init__(self):
        if self.init_from == "scratch":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{self.run_name}--{timestamp}"
            self.logdir = os.path.join("logs", self.run_id)
            os.makedirs("logs", exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)
            self.model_save_fname = os.path.join(self.logdir, self.run_name+".pt")

class Scheduler:
    def __init__(self, config):
        pass

    def __call__(self, it):
        pass

class Trainer:
    def __init__(self, config, datamodule, model, init_train_info: dict|None = None):
        self.config = config
        self.datamodule = datamodule
        self.model = model
        self.init_train_info = init_train_info
        self.writer = SummaryWriter(log_dir=config.logdir)
        self.device = torch.device(config.device)

    def to_device(self, batch):
        return [a.to(self.device) for a in batch]

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.config.lr)
        if self.init_train_info is not None:
            self.optimizer.load_state_dict(self.init_train_info["opt_state_dict"])
        self.scheduler = None

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
            **{n: {"loss": {"train": [], "eval": []},
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
        n_dataloader = len(self.datamodule.train_dataloader)

        accum_steps = self.config.grad_accum_steps
        self.optimizer.zero_grad()

        for step_num, batch in enumerate(self.datamodule.train_dataloader):
            batch = self.to_device(batch)
            out, loss = self.model(*batch[:-1], batch[-1])

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            B = batch[-1].size(0)
            num_instances += B
            global_step = epoch_num * n_dataloader + step_num

            true_loss = loss.item()
            epoch_history["loss"] += true_loss * B
            self.history["perstep"]["loss"]["train"].append(true_loss)

            step_metrics = self.compute_metrics(out, batch[-1])
            for k, v in step_metrics.items():
                self.history["perstep"][k]["train"].append(v)
                epoch_history[k] += v * B

            # gradient accumulation
            is_last_batch = (step_num + 1) == n_dataloader
            if (step_num + 1) % accum_steps == 0 or is_last_batch:
                if self.scheduler is not None:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.scheduler(global_step)

                # gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

        for k, v in epoch_history.items():
            self.history["perepoch"][k]["train"].append(v / num_instances)

    def eval_step(self, epoch_num):
        self.model.eval()

        num_instances = 0
        epoch_history  = {k: 0.0 for k in self.history["perstep"]}

        for step_num, batch in enumerate(self.datamodule.eval_dataloader):
            batch = self.to_device(batch)
            with torch.no_grad():
                out, loss = self.model(*batch[:-1], batch[-1])
            step_metrics = self.compute_metrics(out, batch[-1])

            B = batch[-1].size(0)
            num_instances += B
            epoch_history["loss"] += loss.item() * B

            global_step =  epoch_num * len(self.datamodule.eval_dataloader) + step_num
            for k,v in step_metrics.items():
                self.history["perstep"][k]["eval"].append(v)
                epoch_history[k] += v * B

            self.history["perstep"]["loss"]["eval"].append(loss.item())

        for k,v in epoch_history.items():
            self.history["perepoch"][k]["eval"].append(v / num_instances)

        log_str = " | ".join([f"train_{k}: {v['train'][-1]:.5f} | eval_{k}: {v['eval'][-1]:.5f}" for k, v in self.history["perepoch"].items()])
        log_str += f" | lr: {self.scheduler(epoch_num):.7f}" if self.scheduler is not None else ""

        if self.config.save_checkpoint and epoch_num > 0 and self.history["perstep"]["loss"]["eval"][-1] < self.best_eval_loss:
            self.best_eval_loss = self.history["perstep"]["loss"]["eval"][-1]
            checkpoint = {
                "epoch_num": epoch_num,
                "best_eval_loss": self.best_eval_loss,
                "train_config": dataclasses.asdict(self.config),
                "model_config": dataclasses.asdict(self.model.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "history": self.history
            }
            log_str += f" | save checkpoint at {self.config.logdir}"
            torch.save(checkpoint, os.path.join(self.config.logdir, "ckpt.pt"))

        print(f"Epoch: {epoch_num:03d} | {log_str}")

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

