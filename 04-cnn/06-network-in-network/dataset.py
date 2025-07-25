import os
import glob
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np

class ImageNet100Dataset(torch.utils.data.Dataset):
    def __init__(self, root="/home/malick/datasets/imagenet-100", train:bool=True, transforms=None):
        with open(os.path.join(root, "Labels.json"), "r") as f:
            labels = json.load(f)

        self.labels = {k: (v,i) for i, (k,v) in enumerate(labels.items())}
        self.num_classes = len(self.labels)
        self.labels_idx2str = {v[1]: v[0] for v in self.labels.values()}
        set_name = "train" if train else "val"
        self.files = glob.glob(f"{root}/{set_name}*/*/*.JPEG", recursive=True)

        self.transforms = transforms

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[path.split("/")[-2]][1]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self): return len(self.files)

    def __str__(self): return "ImageNet100Dataset"
    def __repr__(self): return "ImageNet100Dataset"

class ImageNet100:
    def __init__(self, root="/home/malick/datasets/imagenet-100", batch_size=128, num_workers=4, transforms=None):
        self.train = ImageNet100Dataset(train=True, root=root, transforms=transforms)
        eval_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.eval = ImageNet100Dataset(train=False, root=root, transforms=eval_transforms)

        self.labels = self.train.labels
        self.num_classes = self.train.num_classes
        self.labels_idx2str = self.train.labels_idx2str
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataloader = self.get_dataloader(train=True)
        self.eval_dataloader = self.get_dataloader(train=False)

    def get_dataloader(self, train:bool):
        return torch.utils.data.DataLoader(
            dataset=self.train if train else self.eval, shuffle=train,
            batch_size=self.batch_size, num_workers=self.num_workers)

    def show_images(self, images, labels):
        fig, axes = plt.subplots(4, 5, figsize=(19.8,10.2))
        axes = axes.ravel()
        for i in range(len(axes)):
            axes[i].imshow(images[i], cmap="gray")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(labels[i].split(",")[0], fontsize=10)
        plt.show()

class FancyPCA:
    def __init__(self):
        with open("imagenet-100-pca.json", "r") as f:
            pca_data = json.load(f)
        self.eigvecs = torch.tensor(pca_data["eigvecs"])
        self.eigvals = torch.tensor(pca_data["eigvals"])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        alphas = torch.zeros(size=(3,)).normal_(mean=0., std=0.1)
        rgb = (self.eigvecs @ (alphas * self.eigvals)).view(3, 1, 1)
        img += rgb
        return torch.clamp(img, 0., 1.)


if __name__=="__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=256),
        torchvision.transforms.ToTensor(),
        FancyPCA(),
    ])
    data = ImageNet100(transforms=transforms, batch_size=32)
    for batch in data.eval_dataloader:
        x, y = batch
        labels = [data.labels_idx2str[yi] for yi in y.numpy()]
        data.show_images(x.permute(0,2,3,1)[:20], labels)
        print(x.shape); break
