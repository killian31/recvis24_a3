import torch.nn as nn
import torch.nn.functional as F

nclasses = 500


class CustomCNN(nn.Module):
    def __init__(self, num_classes=nclasses):
        super(CustomCNN, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output Layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Output Layer
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes=500):
        super(ResNet, self).__init__()
        self.dropout_rate = 0.3

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dwconv1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pwconv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_conv1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_bn1 = nn.BatchNorm2d(128)
        self.res_conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_bn2 = nn.BatchNorm2d(128)

        self.dwconv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False
        )
        self.bn5 = nn.BatchNorm2d(128)
        self.pwconv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.dwconv1(x)))
        x = F.relu(self.bn4(self.pwconv1(x)))
        x = self.pool2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        residual = x
        x = F.relu(self.res_bn1(self.res_conv1(x)))
        x = self.res_bn2(self.res_conv2(x))
        x += residual
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = F.relu(self.bn5(self.dwconv2(x)))
        x = F.relu(self.bn6(self.pwconv2(x)))
        x = self.pool3(x)

        x = F.relu(self.bn7(self.conv3(x)))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
