import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample

        # Depthwise Separable Convolution Block 1
        self.dwconv1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False
        )  # Depthwise
        self.bn3 = nn.BatchNorm2d(64)
        self.pwconv1 = nn.Conv2d(
            64, 128, kernel_size=1, stride=1, bias=False
        )  # Pointwise
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample

        # Residual Block
        self.res_conv1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_bn1 = nn.BatchNorm2d(128)
        self.res_conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_bn2 = nn.BatchNorm2d(128)

        # Depthwise Separable Convolution Block 2
        self.dwconv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False
        )  # Depthwise
        self.bn5 = nn.BatchNorm2d(128)
        self.pwconv2 = nn.Conv2d(
            128, 256, kernel_size=1, stride=1, bias=False
        )  # Pointwise
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample

        # Final Convolutional Block
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Depthwise Separable Convolution Block 1
        x = F.relu(self.bn3(self.dwconv1(x)))
        x = F.relu(self.bn4(self.pwconv1(x)))
        x = self.pool2(x)

        # Residual Block
        residual = x
        x = F.relu(self.res_bn1(self.res_conv1(x)))
        x = self.res_bn2(self.res_conv2(x))
        x += residual  # Add residual connection
        x = F.relu(x)

        # Depthwise Separable Convolution Block 2
        x = F.relu(self.bn5(self.dwconv2(x)))
        x = F.relu(self.bn6(self.pwconv2(x)))
        x = self.pool3(x)

        # Final Convolutional Block
        x = F.relu(self.bn7(self.conv3(x)))
        x = self.pool4(x)

        # Fully Connected Layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
