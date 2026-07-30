import torch

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

    xT = torch.randn((36, 1, 28, 28))
    eps0 = torch.randn(xT.size(), device=xT.device)
    timesteps = torch.tensor(ddpm.scheduler.T-1, device=xT.device).type(torch.long).unsqueeze(0)
    timesteps = torch.repeat_interleave(timesteps, xT.size(0), 0)

    trainer = Trainer(TrainConfig(run_name="ddpm", max_epochs=700), datamodule, ddpm, xT)
    trainer.fit()
