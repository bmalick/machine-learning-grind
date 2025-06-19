import torch
from torch import nn


class NiNBlock(nn.Sequential):
    def __init__(self, kernel_size, in_channels: int, out_channels: int, stride: int, padding: int):
        super(NiNBlock, self).__init__()

        self.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=0.)
        )
        self.append(nn.ReLU())
        for _ in range(2):
            self.append(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=0.))
            self.append(nn.ReLU())

        self.init()

    def init(self):
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                # torch.nn.init.normal_(tensor=layer.weight, mean=0., std=0.01)
                torch.nn.init.xavier_uniform_(tensor=layer.weight)


class NetworkInNetwork(nn.Sequential):
    """Inspired from AlexNet architecture"""

    def __init__(self, num_classes:int=100):
        super(NetworkInNetwork, self).__init__()

        self.append(NiNBlock(kernel_size=11, in_channels=3, out_channels=96, stride=4, padding=2))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.append(nn.Dropout(0.5))

        self.append(NiNBlock(kernel_size=5, in_channels=96, out_channels=256, stride=1, padding=2))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.append(nn.Dropout(0.5))

        self.append(NiNBlock(kernel_size=3, in_channels=256, out_channels=384, stride=1, padding=1))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.append(nn.Dropout(0.5))

        self.append(NiNBlock(kernel_size=3, in_channels=384, out_channels=512, stride=1, padding=1))
        self.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.append(nn.Dropout(0.5))

        self.append(NiNBlock(kernel_size=3, in_channels=512, out_channels=num_classes, stride=1, padding=1))
        self.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.append(nn.Flatten())

    def layer_summary(self, shape):
        x = torch.randn(shape)
        print("input:".ljust(20), x.shape)
        for layer in self:
            x = layer(x)
            print((layer.__class__.__name__ + ":").ljust(20), x.shape)

if __name__ == "__main__":
    shape = (1, 3, 224, 224)
    model = NetworkInNetwork()
    print("Model:\n", model, "\n")
    model.layer_summary(shape)
