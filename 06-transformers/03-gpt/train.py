import os
import math
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import dataclasses
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    run_name: str = "gpt-pretrained"
    max_iters: int = 600000
    eval_iters: int = 250
    # log_interval: int = 1
    eval_interval: int = 2000
    # lr: float = 1e-3
    weight_decay: float = 1e-2
    min_lr: float = 0.
    max_lr: float = 2.5e-4
    warmup_steps: int = 2000
    grad_clip: float = 1.
    grad_accum_steps: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    init_from: str = "scratch"
    resume_logdir: str|None = None
    run_id: str|None = None
    logdir: str|None = None
    save_checkpoint: bool = True
    save_model: bool = True
    model_save_fname: str|None = None

    plot_interval: int = 100
    figsize: tuple[float, float] = (8., 4.5)

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
        self.alpha = (config.max_lr - config.min_lr) / config.warmup_steps
        self.warmup_steps = config.warmup_steps
        self.max_lr = config.max_lr
        self.min_lr = config.min_lr
        self.max_iters = config.max_iters

    def __call__(self, it):
        if it < self.warmup_steps:
            return self.alpha * it + self.min_lr
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1. + math.cos(math.pi * (it - self.warmup_steps) / (self.max_iters - self.warmup_steps)))

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.config.max_lr, weight_decay=self.config.weight_decay)
        if self.init_train_info is not None:
            self.optimizer.load_state_dict(self.init_train_info["opt_state_dict"])
        self.scheduler = Scheduler(self.config)

    def configure_metrics(self):
        self.metric_names = ["acc"]
        self.metric_funcs = [
            lambda x,y: (x.argmax(dim=-1)==y).float().mean()
        ]

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
            "train_history": {
                "loss": {"train": []},
                "grad_norm": {"train": []},
                **{m: {"train": []} for m in self.metric_names},
            },
            "eval_history": {
                "loss": {"train": [], "eval": []},
                **{m: {"train": [], "eval": []} for m in self.metric_names},
            }
        } if self.init_train_info is None else self.init_train_info["history"]

        iter_num = 0 if self.init_train_info is None else self.init_train_info["iter_num"] + 1
        self.best_eval_loss = 1e9 if self.init_train_info is None else self.init_train_info["best_eval_loss"]

        eval_losses = None
        while True:
            self.train_step(iter_num)

            if iter_num > 0 and iter_num % self.config.eval_interval == 0:
                eval_losses = self.eval_step(iter_num)
                self.make_plots("eval_history")

            if iter_num > 0 and iter_num % self.config.plot_interval == 0:
                self.make_plots("train_history")

            iter_num += 1
            if iter_num >= self.config.max_iters:
                if eval_losses is None or (iter_num - 1) % self.config.eval_interval != 0:
                    eval_losses = self.eval_step(iter_num)
                break

        self.writer.close()
        self.save_model()
        self.save_history()
        for n in ["train_history", "eval_history"]:
            self.make_plots(n)

    def train_step(self, iter_num: int):
        self.model.train()

        accum_steps = self.config.grad_accum_steps
        self.optimizer.zero_grad()
        step_history_sum = {"loss": 0.0, **{k: 0.0 for k in self.metric_names}}

        for _ in range(accum_steps):
            batch = self.datamodule.get_batch("train")
            x, y = self.to_device(batch)
            logits, loss = self.model(x, y)

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            step_history_sum["loss"] += loss.item()
            step_metrics = self.compute_metrics(logits, y)
            for k,v in step_metrics.items():
                step_history_sum[k] += v

        if self.scheduler is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.scheduler(iter_num)

        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
        self.history["train_history"]["grad_norm"]["train"].append(gn.item())
        self.optimizer.step()

        for k,v in step_history_sum.items():
            self.history["train_history"][k]["train"].append(v / accum_steps)

    def eval_step(self, iter_num: int) -> dict[str,dict[str,float]]:
        self.model.eval()

        out = {}
        for split in ["train", "eval"]:
            metrics = {"loss": torch.zeros(self.config.eval_iters),
                       **{m: torch.zeros(self.config.eval_iters) for m in self.metric_names}}

            for k in range(self.config.eval_iters):
                batch = self.datamodule.get_batch(split)
                x, y = self.to_device(batch)

                with torch.no_grad():
                    logits, loss = self.model(x, y)
                metrics["loss"][k] = loss.item()

                step_metrics = self.compute_metrics(logits, y)
                for n,v in step_metrics.items():
                    metrics[n][k] = v

            out[split] = {n: metrics[n].mean().item() for n in metrics}

        for split in ["train", "eval"]:
            for k, v in out[split].items():
                self.history["eval_history"][k][split].append(v)

        log_str = " | ".join([f"train_{k}: {out['train'][k]:.5f} | eval_{k}: {out['eval'][k]:.5f}" for k in out["train"].keys()])
        log_str += f" | lr: {self.scheduler(iter_num):.7f}" if self.scheduler is not None else ""

        if self.config.save_checkpoint and iter_num > 0 and out["eval"]["loss"] < self.best_eval_loss:
            self.best_eval_loss = out["eval"]["loss"]
            checkpoint = {
                "iter_num": iter_num,
                "best_eval_loss": self.best_eval_loss,
                "train_config": dataclasses.asdict(self.config),
                "model_config": dataclasses.asdict(self.model.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "history": self.history
            }
            log_str += f" | save checkpoint at {self.config.logdir}"
            torch.save(checkpoint, os.path.join(self.config.logdir, "ckpt.pt"))

        print(f"Step: {iter_num:06d} | {log_str}")
        return out

    def save_model(self):
        if self.config.save_model:
            torch.save(self.model.state_dict(), self.config.model_save_fname)
            print(f"Model saved at {self.config.model_save_fname}")

    def save_history(self):
        fname = os.path.join(self.config.logdir, "history.json")
        with open(fname, "w") as f:
            json.dump(self.history, f)
        print(f"Save history at {fname}")

    def make_plots(self, plot_name):
        metrics = self.history[plot_name]
        for metric_name, m_values in metrics.items():
            fig, ax = plt.subplots(figsize=self.config.figsize)
            for split, values in m_values.items():
                ls = "-" if split=="train" else "--"
                color = "blue" if split == "train" else "orange"
                if len(values) > 0:
                    ax.plot(values, label=split, linestyle=ls, c=color)
            ax.grid()
            ax.legend()
            ax.set_xlabel("steps")
            ax.set_title(metric_name)
            fig.savefig(os.path.join(self.config.logdir, f"{metric_name}-{plot_name.replace('_', '-')}.jpg"))
            plt.close()

