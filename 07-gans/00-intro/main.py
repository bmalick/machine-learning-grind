#!/home/malick/miniconda3/envs/pt/bin/python3

import torch
from torch import nn

import utils

if __name__=="__main__":
    data = utils.get_linear_data()
    dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(data),
            batch_size=8, shuffle=True)

    linear_generator = nn.Sequential(nn.Linear(2, 2))
    linear_discriminator = nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(),
        nn.Linear(5, 3), nn.Tanh(),
        nn.Linear(3, 1)
    )

    lr_discriminator = 0.05
    lr_generator = 0.005
    latent_dim = 2
    num_epochs = 20
    utils.train_gan(discriminator=linear_discriminator, generator=linear_generator,
                dataloader=dataloader, data=data[:100], num_epochs=num_epochs, lr_generator=lr_generator,
                lr_discriminator=lr_discriminator, latent_dim=latent_dim)
