import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def get_linear_data(n: int = 1000) -> torch.tensor:
    X = torch.normal(0.0, 1, (n, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b
    return data


def update_discriminator(X, Z, discriminator, generator, loss, trainer_discriminator):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)

    trainer_discriminator.zero_grad()

    real_y = discriminator(X)
    fake_X = generator(Z)
    # Do not need to compute gradient for generator, detach it from computing gradients.
    fake_y = discriminator(fake_X.detach())
    loss_discriminator = (loss(real_y, ones.reshape(real_y.shape)) +
                          loss(fake_y, zeros.reshape(fake_y.shape))) / 2
    loss_discriminator.backward()
    trainer_discriminator.step()
    return loss_discriminator

def update_generator(Z, discriminator, generator, loss, trainer_generator):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)

    trainer_generator.zero_grad()

    fake_x = generator(Z)
    fake_y = discriminator(fake_x)
    loss_generator = loss(fake_y, ones.reshape(fake_y.shape))
    loss_generator.backward()
    trainer_generator.step()
    return  loss_generator


def train_gan(discriminator, generator, dataloader, data, num_epochs, lr_generator, lr_discriminator, latent_dim, visualize=False):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in discriminator.parameters(): nn.init.normal_(w, 0., 0.02)
    for w in generator.parameters(): nn.init.normal_(w, 0., 0.02)
    trainer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator)
    trainer_generator = torch.optim.Adam(generator.parameters(), lr=lr_generator)

    metrics = []
    os.makedirs("visualizations", exist_ok=True)
    for epoch in range(num_epochs):
        loss_generator = 0
        loss_discriminator = 0
        num_instances = 0
        for step_num, (X,) in enumerate(dataloader):
            batch_size = X.shape[0]
            num_instances += batch_size
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            loss_discriminator += update_discriminator(X, Z, discriminator, generator, loss, trainer_discriminator).item()
            loss_generator += update_generator(Z, discriminator, generator, loss, trainer_generator).item()
            print(f"[Epoch {epoch+1}/{num_epochs}] [Step {step_num}/{len(dataloader)}] loss_D: {loss_discriminator/num_instances:.5f}, loss_G: {loss_generator/num_instances:.5f}")

        loss_generator /= num_instances
        loss_discriminator /= num_instances
        metrics.append([loss_generator, loss_discriminator])

        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = generator(Z).detach().numpy()
        plt.scatter(data[:, 0], data[:, 1], label="real")
        plt.scatter(fake_X[:, 0], fake_X[:, 1], label="generated")
        plt.legend(["real", "generated"])
        plt.savefig(f"visualizations/{epoch}.png")
        if visualize: plt.show()
        else: plt.close()

    plt.close()

    metrics = np.array(metrics)

    plt.plot(metrics[:, 0], label="generator")
    plt.plot(metrics[:, 1], label="discriminator", linestyle="--")
    plt.legend()
    plt.ylabel("loss")
    plt.show()

