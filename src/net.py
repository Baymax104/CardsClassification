import torch.nn.functional as F
import torch.nn as nn


class ClassifyNet(nn.Module):
    """
    Final Net Structure, its record locates at ../model/05-30_02-00/model-05-30_02-00.pt
    """

    def __init__(self):
        super(ClassifyNet, self).__init__()

        shape = 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, padding=1, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.res_conv = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        shape = self.cal_shape(shape, 0, 2, 2)
        shape = self.cal_shape(shape, 0, 2, 2)

        self.conv1x1_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)

        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm2d(num_features=32)

        self.linear = nn.Linear(32 * shape * shape, 53)


    def forward(self, x):
        out = x

        # cnn
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        # residual
        out1 = self.conv2(out)
        out1 = self.bn2(out1)
        out1 = F.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        out2 = self.res_conv(out)

        out1 += out2
        out1 = F.relu(out1)

        # cnn
        out1 = self.conv4(out1)
        out1 = self.bn4(out1)
        out1 = F.relu(out1)
        out1 = self.pool2(out1)

        # 1x1 convolution
        out1 = self.conv1x1_1(out1)
        out1 = self.bn5(out1)
        out1 = F.relu(out1)

        out1 = self.conv1x1_2(out1)
        out1 = self.bn6(out1)
        out1 = F.relu(out1)

        # flatten
        out1 = out1.view(out1.size(dim=0), -1)

        # linear
        out1 = F.softmax(self.linear(out1), dim=1)
        return out1

    @staticmethod
    def cal_shape(n, padding, kernel_size, stride):
        return int((n + 2 * padding - kernel_size) / stride) + 1
