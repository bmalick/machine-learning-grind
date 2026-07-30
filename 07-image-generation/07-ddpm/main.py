import os
import json
import torch
import matplotlib.pyplot as plt

from data import MnistConfig, Mnist, FermatSpiralData, SpiralDataConfig
from model import DDPMConfig, DDPM, SpiralMLP
from train import TrainConfig, Trainer

torch.manual_seed(14)

if __name__ == "__main__":
    x_T = torch.randn((1000,2))
    datamodule = FermatSpiralData(SpiralDataConfig(n_samples=10000, train_batch_size=256))
    ddpm = SpiralMLP(DDPMConfig(timesteps=100))
    trainer = Trainer(TrainConfig(run_name="ddpm-simple", max_epochs=700), datamodule, ddpm, x_T)
    trainer.fit()

    datamodule = Mnist(MnistConfig())
    ddpm = DDPM(DDPMConfig())

    x0,_ = next(iter(Mnist(MnistConfig(train_batch_size=36)).train_dataloader))
    eps0 = torch.randn(x0.size(), device=x0.device)
    timesteps = torch.tensor(ddpm.scheduler.T-1, device=x0.device).type(torch.long).unsqueeze(0)
    timesteps = torch.repeat_interleave(timesteps, x0.size(0), 0)
    xT = ddpm.scheduler.add_noise(x0, eps0, timesteps)

    trainer = Trainer(TrainConfig(run_name="ddpm", max_epochs=1000), datamodule, ddpm, xT)
    trainer.fit()
