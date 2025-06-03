import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int=100):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            # For (227x227) images
            # nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False), nn.ReLU(),   # (B, 96, 55, 55)

            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False), nn.ReLU(),   # (B, 96, 55, 55)
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),                                                # (B, 96, 55, 55)
            nn.MaxPool2d(kernel_size=3, stride=2),                                                                   # (B, 96, 27, 27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=1.), nn.ReLU(),     # (B, 256, 27, 27)
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),                                                # (B, 256, 27, 27)
            nn.MaxPool2d(kernel_size=3, stride=2),                                                                   # (B, 256, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), # (B, 384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=1.), nn.ReLU(),    # (B, 384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=1.), nn.ReLU(),    # (B, 256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),                                                                   # (B, 256, 6, 6)
            nn.Flatten(),                                                                                            # (B, 9216)
            nn.Linear(in_features=9216, out_features=4096, bias=1.), nn.ReLU(), nn.Dropout(0.5),                     # (B, 4096)
            nn.Linear(in_features=4096, out_features=4096, bias=1.), nn.ReLU(), nn.Dropout(0.5),                     # (B, 4096)
            nn.Linear(in_features=4096, out_features=num_classes, bias=1.)                                           # (B, num_classes)
        )

        self.init()


    def init(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.normal_(tensor=layer.weight, mean=0., std=0.01)

    def forward(self, x):
        return self.net(x)



if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = AlexNet()
    output = model(x)
    print("Model:\n", model, "\n")
    print("x:", x.shape)
    print("output:", output.shape)
