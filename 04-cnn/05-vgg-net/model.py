import torch
from torch import nn


class VGGBlock(nn.Sequential):
    def __init__(self, num_convs: int, init_channels: int, out_channels: int):
        super(VGGBlock, self).__init__()

        self.append(
            nn.Conv2d(in_channels=init_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=0.)
        )
        self.append(nn.ReLU())
        for _ in range(num_convs-1):
            self.append(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=0.)
            )
            self.append(nn.ReLU())
        self.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.init()

    def init(self):
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                # torch.nn.init.normal_(tensor=layer.weight, mean=0., std=0.01)
                torch.nn.init.xavier_uniform_(tensor=layer.weight)


class VGGNet(nn.Sequential):
    _architecture = {
        "vgg16": [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
        "vgg19": [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)],
    }

    def __init__(self, image_channels:int=3, arch:str="vgg16", num_classes:int=100):
        assert arch in ["vgg16", "vgg19"], "arch must be vgg16 or vgg19"
        super(VGGNet, self).__init__()

        self.arch_name = arch
        architecture = self._architecture[arch]
        for i, (num_convs, out_channels) in enumerate(architecture):
            in_channels = image_channels if i==0 else architecture[i-1][1]
            self.append(
                VGGBlock(num_convs=num_convs, init_channels=in_channels, out_channels=out_channels))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096, bias=0.), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=0.), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes, bias=0.)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # torch.nn.init.normal_(tensor=layer.weight, mean=0., std=0.01)
                torch.nn.init.xavier_uniform_(tensor=layer.weight)

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = VGGNet(arch="vgg16")
    print("Model:\n", model, "\n")
    output = model(x)
    print("x:", x.shape)
    print("output:", output.shape)
