import os
import torch
import matplotlib.pyplot as plt

from data import show_images
from model import DDPMConfig, DDPM, SpiralMLP, ddpm_inference

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logdir = "./logs/ddpm-simple--20260713-203557"
    ckpt = torch.load(os.path.join(logdir, "ddpm-simple.pt"), map_location=device)
    ddpm = SpiralMLP(DDPMConfig(timesteps=100))
    ddpm.load_state_dict(ckpt)
    xT = torch.randn((1000,2))
    ddpm.eval()
    reconstructed = ddpm_inference(ddpm, xT, False)
    reconstructed = reconstructed.detach().cpu().numpy()
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], s=5)
    plt.savefig(os.path.join(logdir, f"inference.jpg"))
    plt.show()

    logdir = "./logs/ddpm--20260730-004143"

    ddpm = DDPM(DDPMConfig()).to(device)
    ckpt = torch.load(os.path.join(logdir, "ddpm.pt"), map_location=device)
    ddpm.load_state_dict(ckpt)
    ddpm.eval()
    xT = torch.randn((36, 1, 28, 28), device=device)
    eps0 = torch.randn(xT.size(), device=xT.device)
    timesteps = torch.tensor(ddpm.scheduler.T-1, device=xT.device).type(torch.long).unsqueeze(0)
    timesteps = torch.repeat_interleave(timesteps, xT.size(0), 0)
    reconstructed, ims = ddpm_inference(ddpm, xT, True)

    os.makedirs(os.path.join(logdir, "inferences"), exist_ok=True)
    for i in range(len(ims)):
        t = len(ims) - i
        show_images(ims[i], nrow=12, figsize=(19.2,10.8), show=False, save_name=os.path.join(logdir, "inferences", f"inference-{i}.jpg"), white_bg=True, title=f"step {t}")

    show_images(reconstructed, nrow=12, figsize=(19.2,10.8), show=True, save_name=os.path.join(logdir, f"inference.jpg"), white_bg=True)



