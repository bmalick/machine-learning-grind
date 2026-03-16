import torchvision
from sklearn.decomposition import IncrementalPCA
import json
import time
from tqdm import tqdm

import dataset

def pca_imagenet_100(num_labels: int = None, batch_size: int = 128):
    start = time.time()


    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=256),
        torchvision.transforms.ToTensor(),
    ])
    data = dataset.ImageNet100(transforms=transforms, batch_size=batch_size, num_labels=num_labels)

    ipca = IncrementalPCA(n_components=3)

    for images, _ in tqdm(data.train_dataloader):
        batch = images.permute(0, 2, 3, 1).reshape(-1, 3)
        ipca.partial_fit(batch.numpy())

    eigvals = ipca.explained_variance_
    eigvecs = ipca.components_.T

    with open("imagenet-100-pca.json", "w") as f:
        json.dump({
            "eigvecs": eigvecs.tolist(),
            "eigvals": eigvals.tolist(),
        }, f)

    duration = time.time() - start
    print(f"PCA took {duration:.5f} seconds")

    print("PCA eigvals and eigvecs on Imagenet-100 are saved into imagenet-100-pca.json")

if __name__ == "__main__":
    # pca_imagenet_100(batch_size=256)
    pca_imagenet_100(batch_size=256, num_labels=10)
