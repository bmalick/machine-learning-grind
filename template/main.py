import os
import torch

from data import DataConfig, DataModule
from model import ModuleConfig, Module
from train import TrainConfig, Trainer

torch.manual_seed(42)

def init_from(config):
    if config.init_from == "scratch":
        print("Training from scratch")
        return Module(ModuleConfig()), None
    elif config.init_from == "resume":
        print(f"Training from resume: {config.resume_logdir}")
        if not os.path.exists(config.resume_logdir):
            print(f"{config.resume_logdir} does not exist.")
            return
        ckpt_fname = os.path.join(config.resume_logdir, "ckpt.pt")
        checkpoint = torch.load(ckpt_fname, map_location=config.device)
        model = Module(TrainConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"])
        config.run_id = checkpoint["train_config"]["run_id"]
        config.logdir = checkpoint["train_config"]["logdir"]
        config.model_save_name = checkpoint["train_config"]["model_save_fname"]
        iter_num = checkpoint["iter_num"]
        print(f"Training from resume: {config.resume_logdir}. Last step {iter_num}")
        return model, {"iter_num": iter_num, "best_eval_loss": checkpoint["best_eval_loss"],
                     "opt_state_dict": checkpoint["optimizer"], "history": checkpoint["history"]}
    else:
        print(f"{config.init_from} unknown. Please use scratch or resume")
        return


if __name__ == "__main__":
    datamodule = DataModule(DataConfig(train_batch_size=4, eval_batch_size=4))
    train_cfg = TrainConfig(resume_logdir="./logs/gpt-pretrained--20260706-093202", init_from="resume")
    init_out = init_from(train_cfg)
    if init_out is not None:
        model, init_train_info = init_out
        trainer = Trainer(train_cfg, datamodule, model, init_train_info)
        trainer.fit()
