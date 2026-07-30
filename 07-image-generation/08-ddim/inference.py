import os
import torch
import matplotlib.pyplot as plt

from model import DDPMConfig, DDPM, SpiralMLP, ddpm_inference, ddim_inference
import torchvision
import torchvision.transforms as T

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logdir = "./logs/ddpm-simple--20260713-203557"
    # ckpt = torch.load(os.path.join(logdir, "ddpm-simple.pt"), map_location=device)
    # ddpm = SpiralMLP(DDPMConfig(timesteps=100))
    # ddpm.load_state_dict(ckpt)
    # xT = torch.randn((1000,2))
    # ddpm.eval()
    #
    # fig, axes = plt.subplots(2,2, figsize=(10,8))
    # axes = axes.ravel()
    #
    # reconstructed = ddpm_inference(ddpm, xT)
    # reconstructed = reconstructed.detach().cpu().numpy()
    # axes[0].scatter(reconstructed[:, 0], reconstructed[:, 1], s=5)
    # axes[0].set_title(f"DDPM sampling: {ddpm.scheduler.T} steps")
    # axes[0].axis("off")
    #
    # num_steps = [5, 10, 20]
    # for i in range(len(num_steps)):
    #     n_steps = num_steps[i]
    #     reconstructed = ddim_inference(ddpm, xT, n_steps)
    #     reconstructed = reconstructed.detach().cpu().numpy()
    #     axes[i+1].scatter(reconstructed[:, 0], reconstructed[:, 1], s=5)
    #     axes[i+1].set_title(f"DDIM sampling: {n_steps} steps")
    #     axes[i+1].axis("off")
    #
    #
    # fig.savefig(os.path.join(logdir, f"inference.jpg"))
    # plt.show()

    logdir = "./logs/ddpm--20260730-004143"

    ddpm = DDPM(DDPMConfig()).to(device)
    ckpt = torch.load(os.path.join(logdir, "ddpm.pt"), map_location=device)
    ddpm.load_state_dict(ckpt)
    ddpm.eval()

    xT = torch.randn((16, 1, 28, 28), device=device)
    eps0 = torch.randn(xT.size(), device=xT.device)
    timesteps = torch.tensor(ddpm.scheduler.T-1, device=xT.device).type(torch.long).unsqueeze(0)
    timesteps = torch.repeat_interleave(timesteps, xT.size(0), 0)
    reconstructed = ddpm_inference(ddpm, xT)

    fig, axes = plt.subplots(2,3, figsize=(19.2, 10.8))
    axes = axes.ravel()

    reconstructed = ddpm_inference(ddpm, xT)
    reconstructed = reconstructed.detach().cpu()
    to_img = lambda x: T.ToPILImage()(torchvision.utils.make_grid(torch.clamp((x + 1.) / 2., 0., 1.), nrow=4))

    axes[0].imshow(to_img(reconstructed))
    axes[0].set_title(f"DDPM sampling: {ddpm.scheduler.T} steps")
    axes[0].axis("off")

    num_steps = [5, 10, 50, 100, 200]
    for i in range(len(num_steps)):
        n_steps = num_steps[i]
        reconstructed = ddim_inference(ddpm, xT, n_steps)
        reconstructed = reconstructed.detach().cpu()
        axes[i+1].imshow(to_img(reconstructed))
        axes[i+1].set_title(f"DDIM sampling: {n_steps} steps")
        axes[i+1].axis("off")

    fig.savefig(os.path.join(logdir, f"inference.jpg"))
    # plt.show()
    plt.close()
