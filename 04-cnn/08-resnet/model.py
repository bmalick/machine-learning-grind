import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, channels: int, stride:int=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels
        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=channels, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=channels)

        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

        if stride!=1 or in_channels!=out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else: self.downsample = None
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return F.relu(y)

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, channels: int, stride:int=1):
        super(BottleNeck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=channels)
        self.bn1 = nn.BatchNorm2d(num_features=channels)

        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=channels, out_channels=channels, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

        self.conv3 = nn.Conv2d(kernel_size=1, in_channels=channels, out_channels=out_channels)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        if stride!=1 or in_channels!=out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else: self.downsample = None
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return F.relu(y)

class ResNetBlock(nn.Sequential):
    def __init__(self, in_channels: int, num_blocks: int, channels: int, stride: int=1, expansion: int=1):
        super(ResNetBlock, self).__init__()

        self.append(
            BasicBlock(in_channels=in_channels, channels=channels, stride=stride)
            if expansion==1 else
            BottleNeck(in_channels=in_channels, channels=channels, stride=stride)
        )
        for _ in range(num_blocks-1):
            self.append(
                BasicBlock(in_channels=channels*expansion, channels=channels, stride=1)
                if expansion==1 else
                BottleNeck(in_channels=channels*expansion, channels=channels, stride=1)
            )

class ResNet(nn.Sequential):
    def __init__(self,
                 arch: list,
                 expansion: int = 1,
                 in_channels: int = 3,
                 num_classes: int = 100):
        super(ResNet, self).__init__()

        init_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=7, in_channels=in_channels, out_channels=64, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        current_channels = 64
        for i, num_blocks in enumerate(arch):
            stride = 1 if i==0 else 2
            block_name = f"conv{i+2}_x"
            block_channels = init_channels * (2**i) # 64, 128, 256, 512
            self.add_module(
                    block_name, 
                    ResNetBlock(in_channels=current_channels, num_blocks=num_blocks, channels=block_channels, stride=stride, expansion=expansion))
            current_channels = block_channels * expansion

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=current_channels, out_features=num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)



class ResNet18(ResNet):
    def __init__(self, in_channels: int=3, num_classes=100):
        super().__init__(
            arch=[2,2,2,2], expansion=1,
            in_channels=in_channels, num_classes=num_classes)


class ResNet34(ResNet):
    def __init__(self, in_channels: int=3, num_classes=100):
        super().__init__(
            arch=[3,4,6,3], expansion=1,
            in_channels=in_channels, num_classes=num_classes)


class ResNet50(ResNet):
    def __init__(self, in_channels: int=3, num_classes=100):
        super().__init__(
            arch=[3,4,6,3], expansion=4,
            in_channels=in_channels, num_classes=num_classes)


class ResNet101(ResNet):
    def __init__(self, in_channels: int=3, num_classes=100):
        super().__init__(
            arch=[3,4,23,3], expansion=4,
            in_channels=in_channels, num_classes=num_classes)

class ResNet152(ResNet):
    def __init__(self, in_channels: int=3, num_classes=100):
        super().__init__(
            arch=[3,8,36,3], expansion=4,
            in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    shape = (1, 3, 224, 224)
    x = torch.randn(shape)

    model = ResNet18(in_channels=3, num_classes=100)
    # model = ResNet34(in_channels=3, num_classes=100)
    # model = ResNet50(in_channels=3, num_classes=100)
    # model = ResNet101(in_channels=3, num_classes=100)
    # model = ResNet152(in_channels=3, num_classes=100)

    print("Model:\n", model, "\n")
    out = model(x)
    print("input shape:", x.shape)
    print("final output shape:", out.shape)
