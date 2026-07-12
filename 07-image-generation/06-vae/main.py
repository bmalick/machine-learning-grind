import torch

torch.manual_seed(14)

from data import DataModule, DataConfig
from model import VariationalAutoEncoderMLP, VariationalAutoEncoderConv, ModuleConfig
from train import TrainConfig, Trainer

import os
import json
import matplotlib.pyplot as plt

# def load_checkpoint(logdir: str, device="cpu"):
#     ckpt_fname = os.path.join(logdir, "ckpt.pt")
#     checkpoint = torch.load(ckpt_fname, map_location=device)
#     return checkpoint["train_config"]["run_id"], checkpoint["history"]

def load_checkpoint(logdir: str):
    run_id = os.path.basename(logdir).split("--")[0]
    history = json.load(open(os.path.join(logdir, "history.json"), "r"))
    return run_id, history

def compare_plots(logdir1, logdir2, figsize=(8., 4.5)):
    run_id1, history1 = load_checkpoint(logdir1)
    run_id2, history2 = load_checkpoint(logdir2)
    for pername in ["perstep", "perepoch"]:
        for loss_name in history1[pername]:
            fig, ax = plt.subplots(figsize=figsize)
            for (split1,value1), (split2,value2) in zip(history1[pername][loss_name].items(), history2[pername][loss_name].items()):
                if value1:
                    ax.plot(value1, label=f"{run_id1} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                if value2:
                    ax.plot(value2, label=f"{run_id2} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
            ax.grid()
            ax.legend()
            ax.set_xlabel(pername.replace("per","")+"s")
            ax.set_title(loss_name)
            compare_dir = f"{run_id1}-vs-{run_id2}"
            os.makedirs(compare_dir, exist_ok=True)
            fig.savefig(os.path.join(compare_dir, f"{pername}-{loss_name}.jpg"))
            plt.close()

if __name__ == "__main__":
    datamodule = DataModule(DataConfig())
    fixed_batch = next(iter(DataModule(DataConfig(train_batch_size=36)).train_dataloader))[0]

    auto_enc_mlp = VariationalAutoEncoderMLP(ModuleConfig(hidden_dim=500))
    mlp_trainer = Trainer(TrainConfig(run_name="mlp-vae"), datamodule, auto_enc_mlp, fixed_batch)
    mlp_trainer.fit()

    auto_enc_conv = VariationalAutoEncoderConv(ModuleConfig(hidden_dim=32))
    conv_trainer = Trainer(TrainConfig(run_name="conv-vae"), datamodule, auto_enc_conv, fixed_batch)
    conv_trainer.fit()

    compare_plots(mlp_trainer.config.logdir, conv_trainer.config.logdir)


