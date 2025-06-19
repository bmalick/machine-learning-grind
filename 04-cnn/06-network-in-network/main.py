#!/home/malick/miniconda3/envs/pt/bin/python3

import torch
import torchvision
import matplotlib.pyplot as plt

import dataset
import model
import train


def main():
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    max_epochs = 90
    log_every = 500

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=256),
        torchvision.transforms.RandomCrop(size=(224,224)),
        torchvision.transforms.ToTensor(),
        dataset.FancyPCA(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = dataset.ImageNet100(batch_size=batch_size, transforms=transforms)
    num_classes = data.num_classes


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg_net = model.NetworkInNetwork(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(lr=learning_rate, params=vgg_net.parameters(), momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1)

    overall_metrics = train.train(
        device=device,
        model=vgg_net,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        log_every=log_every,
    )
    for set_name in ["train", "eval"]:
        for metric in ["loss", "accuracy"]:
            plt.plot(overall_metrics[f"{set_name}_step"], overall_metrics[f"{set_name}_{metric}"])
            plt.xlabel("steps")
            plt.ylabel(f"{metric}")
            plt.title(f"{set_name} {metric}")
            plt.show(block=False)
            plt.pause(5)
            plt.savefig(f"{set_name}-{metric}.png")
            plt.close()

if __name__=="__main__":
    main()
