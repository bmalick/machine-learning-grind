import os
import torch

from data import DataConfig, TinyStoriesData
from model import GPTConfig, GPTPretrained
from train import TrainConfig, Trainer

torch.manual_seed(13)

DATA_CONFIF_ARGS = {
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "num_workers": 2,
    "seq_len": 256
}
MODEL_CONFIG_ARGS = {
    "embed_dim": 512,
    "num_heads": 8,
    "dropout": 0.1,
    "ff_dim": 2048,
    "num_blocks": 6,
    "max_len": 256,
    "vocab_size": 50257
}
TRAIN_CONFIG_ARGS = {
    "run_name": "gpt-pretrained",
    "max_iters": 600000,
    "eval_iters": 250,
    "eval_interval": 2000,
    "weight_decay": 1e-2,
    "min_lr": 0.,
    "max_lr": 2.5e-4,
    "warmup_steps": 2000,
    "grad_clip": 1.,
    "grad_accum_steps": 4,
    "device": "cuda",
    "init_from": "scratch",
    "resume_logdir": None,
    "run_id": None,
    "logdir": None,
    "save_checkpoint": True,
    "save_model": True,
    "model_save_fname": None,
    "plot_interval": 100,
    "figsize": (8., 4.5),
}

def load_checkpoint(fname: str, device):
    return torch.load(fname, map_location=device)


def init_from(config):
    if config.init_from == "scratch":
        print("Training from scratch")
        return GPTPretrained(GPTConfig(**MODEL_CONFIG_ARGS)), None
    elif config.init_from == "resume":
        if not os.path.exists(config.resume_logdir):
            print(f"{config.resume_logdir} does not exist.")
            return
        ckpt_fname = os.path.join(config.resume_logdir, "ckpt.pt")
        checkpoint = torch.load(ckpt_fname, map_location=config.device)
        gpt = GPTPretrained(GPTConfig(**checkpoint["model_config"]))
        gpt.load_state_dict(checkpoint["model"])
        config.run_id = checkpoint["train_config"]["run_id"]
        config.logdir = checkpoint["train_config"]["logdir"]
        config.model_save_name = checkpoint["train_config"]["model_save_fname"]
        iter_num = checkpoint["iter_num"]
        print(f"Training from resume: {config.resume_logdir}. Last step {iter_num}")
        return gpt, {"iter_num": iter_num, "best_eval_loss": checkpoint["best_eval_loss"],
                     "opt_state_dict": checkpoint["optimizer"], "history": checkpoint["history"]}
    else:
        print(f"{config.init_from} unknown. Please use scratch or resume")
        return


if __name__ == "__main__":
    datamodule = TinyStoriesData(DataConfig(**DATA_CONFIF_ARGS))
    train_cfg = TrainConfig(resume_logdir="./logs/gpt-pretrained--20260708-024312/", init_from="resume")
    # train_cfg = TrainConfig(**TRAIN_CONFIG_ARGS)
    init_out = init_from(train_cfg)
    if init_out is not None:
        gpt, init_train_info = init_out
        trainer = Trainer(train_cfg, datamodule, gpt, init_train_info)
        trainer.fit()
