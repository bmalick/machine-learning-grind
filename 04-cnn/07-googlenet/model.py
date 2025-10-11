import torch
from torch import nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, branches, in_channels):
        super(Inception, self).__init__()

        self.branch1 = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=branches[0])
        self.branch2_1 = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=branches[1][0])
        self.branch2_2 = nn.Conv2d(kernel_size=3, padding=1, in_channels=branches[1][0], out_channels=branches[1][1])
        self.branch3_1 = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=branches[2][0])
        self.branch3_2 = nn.Conv2d(kernel_size=5, padding=2, in_channels=branches[2][0], out_channels=branches[2][1])
        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=branches[3])

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2_2(F.relu(self.branch2_1(x))))
        b3 = F.relu(self.branch3_2(F.relu(self.branch3_1(x))))
        b4 = F.relu(self.branch4_2(self.branch4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes:int=100, clf_dropout:float=0.4):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(kernel_size=7, stride=2, padding=3, in_channels=3, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layernorm1 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64)
        self.conv3 = nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layernorm2 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)

        self.inception_3a = Inception(in_channels=192, branches=[64, (96, 128), (16, 32), 32])
        self.inception_3b = Inception(in_channels=256, branches=[128, (128, 192), (32, 96), 64])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = Inception(in_channels=480, branches=[192, (96, 208), (16, 48), 64])
        self.inception_4b = Inception(in_channels=512, branches=[160, (112, 224), (24, 64), 64])
        self.inception_4c = Inception(in_channels=512, branches=[128, (128, 256), (24, 64), 64])
        self.inception_4d = Inception(in_channels=512, branches=[112, (144, 288), (32, 64), 64])
        self.inception_4e = Inception(in_channels=528, branches=[256, (160, 320), (32, 128), 128])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.classifier1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=3),
            nn.Conv2d(kernel_size=1, in_channels=512, out_channels=128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
            nn.Dropout(clf_dropout),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=3),
            nn.Conv2d(kernel_size=1, in_channels=528, out_channels=128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
            nn.Dropout(clf_dropout),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        self.inception_5a = Inception(in_channels=832, branches=[256, (160, 320), (32, 128), 128])
        self.inception_5b = Inception(in_channels=832, branches=[384, (192, 384), (48, 128), 128])

        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=1)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

        self.init_weights()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(self.layernorm1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.maxpool2(self.layernorm2(out))

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.maxpool3(out)

        out = self.inception_4a(out)
        out1 = self.classifier1(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out2 = self.classifier2(out)
        out = self.inception_4e(out)
        out = self.maxpool4(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.flatten(self.dropout(self.avg_pool(out)))
        out = self.linear(out)
        return out, out1, out2

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == "__main__":
    shape = (1, 3, 224, 224)
    x = torch.randn(shape)
    model = GoogLeNet()
    print("Model:\n", model, "\n")
    out, out1, out2 = model(x)
    print("input shape:", x.shape)
    print("output1 shape:", out1.shape)
    print("output2 shape:", out2.shape)
    print("final output shape:", out.shape)
