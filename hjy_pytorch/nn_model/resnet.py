import torch
import torch.nn as nn

"""
    经典ResNet网络实现
"""

""" tips：当卷积层后面跟着BN层时,卷积层不需要bias """

def conv1x1(in_channels, out_channels, stride=1):
    """ 返回给定输入和输出通道的1x1卷积层 """
    return nn.Conv2d(in_channels, out_channels, stride=stride,
                     kernel_size=1, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    """ 返回给定输入和输出通道的3x3卷积层(padding=1) """
    return nn.Conv2d(in_channels, out_channels, stride=stride,
                     kernel_size=3, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ 实现常规残差模块 """
    expansion = 1

    def __init__(self, in_channels, basic_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.shortcut = nn.Sequential()
        if in_channels != basic_channels:
            stride = 2
            self.shortcut.add_module("conv", conv3x3(in_channels, basic_channels, stride))
            self.shortcut.add_module("bn", nn.BatchNorm2d(basic_channels))
        self.relu = nn.ReLU(inplace=True)

        self.res_path = nn.Sequential(
            conv3x3(in_channels, basic_channels, stride),
            nn.BatchNorm2d(basic_channels),
            nn.ReLU(inplace=True),
            conv3x3(basic_channels, basic_channels),
            nn.BatchNorm2d(basic_channels)
        )

    def forward(self, x):
        res_out = self.res_path(x)
        x_hat = self.shortcut(x)
        return self.relu(res_out + x_hat)


class Bottleneck(nn.Module):
    """ 实现瓶颈结构残差模块 """
    expansion = 4

    def __init__(self, in_channels, basic_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != basic_channels * self.expansion:
            self.shortcut.add_module(
                "conv", conv1x1(in_channels, basic_channels * self.expansion, stride=stride)
            )
            self.shortcut.add_module("bn", nn.BatchNorm2d(basic_channels * self.expansion))

        self.res_path = nn.Sequential(
            conv1x1(in_channels, basic_channels),
            nn.BatchNorm2d(basic_channels),
            nn.ReLU(inplace=True),
            conv3x3(basic_channels, basic_channels, stride),
            nn.BatchNorm2d(basic_channels),
            nn.ReLU(inplace=True),
            conv1x1(basic_channels, basic_channels * self.expansion),
            nn.BatchNorm2d(basic_channels * self.expansion),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.res_path(x)
        x_hat = self.shortcut(x)
        return self.relu(res_out + x_hat)


def make_layers(block_class, blocks_num, in_channels, basic_channels, stride=1):
    layers = []
    layers.append(block_class(in_channels, basic_channels, stride))
    in_channels = basic_channels * block_class.expansion
    for _ in range(1, blocks_num):
        layers.append(block_class(in_channels, basic_channels))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ 实现ResNet网络 """

    def __init__(self, block_class, blocks_nums, in_channels=3, classes_num=10,
                 zero_init_residual=False):
        super(ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = make_layers(block_class, blocks_nums[0], 64, 64)
        self.layer2 = make_layers(block_class, blocks_nums[1], 64 * block_class.expansion, 128, 2)
        self.layer3 = make_layers(block_class, blocks_nums[2], 128 * block_class.expansion, 256, 2)
        self.layer4 = make_layers(block_class, blocks_nums[3], 256 * block_class.expansion, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_class.expansion, classes_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (Bottleneck, BasicBlock)):
                    nn.init.constant_(m.res_path[-1].weight, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = torch.flatten(self.avgpool(x), 1)
        out = self.fc(features)

        return out


def resnet18(in_channels=3, classes_num=10, **kwargs):
    """ 实现经典ResNet-18网络 """
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, classes_num, **kwargs)


def resnet50(in_channels=3, classes_num=10, **kwargs):
    """ 实现经典ResNet-50网络 """
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, classes_num, **kwargs)


""" 官方实现和本人实现的ResNet网络在CIFAR10数据集上分类效果对比"""
# 参数：learning_rate = 0.001，Adam优化算法（默认参数），交叉熵损失函数

#  resnet_18 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0054, train accuracy:0.501, test accuracy:0.583, time:9.56s
# epoch:2, train loss:0.0039, train accuracy:0.643, test accuracy:0.665, time:9.64s
# epoch:3, train loss:0.0032, train accuracy:0.707, test accuracy:0.691, time:9.66s
# epoch:4, train loss:0.0027, train accuracy:0.753, test accuracy:0.703, time:9.71s
# epoch:5, train loss:0.0023, train accuracy:0.789, test accuracy:0.724, time:9.70s
# epoch:6, train loss:0.0020, train accuracy:0.823, test accuracy:0.724, time:9.73s
# epoch:7, train loss:0.0016, train accuracy:0.856, test accuracy:0.731, time:9.75s
# epoch:8, train loss:0.0013, train accuracy:0.880, test accuracy:0.730, time:9.76s
# epoch:9, train loss:0.0010, train accuracy:0.907, test accuracy:0.738, time:9.78s
# epoch:10, train loss:0.0009, train accuracy:0.921, test accuracy:0.722, time:9.78s
#
# end...
#
#  my_resnet_18 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0054, train accuracy:0.509, test accuracy:0.589, time:13.74s
# epoch:2, train loss:0.0039, train accuracy:0.649, test accuracy:0.652, time:13.77s
# epoch:3, train loss:0.0032, train accuracy:0.714, test accuracy:0.688, time:13.78s
# epoch:4, train loss:0.0027, train accuracy:0.761, test accuracy:0.706, time:13.86s
# epoch:5, train loss:0.0022, train accuracy:0.800, test accuracy:0.720, time:13.87s
# epoch:6, train loss:0.0019, train accuracy:0.832, test accuracy:0.728, time:13.85s
# epoch:7, train loss:0.0016, train accuracy:0.856, test accuracy:0.740, time:13.89s
# epoch:8, train loss:0.0013, train accuracy:0.884, test accuracy:0.737, time:13.86s
# epoch:9, train loss:0.0010, train accuracy:0.908, test accuracy:0.740, time:13.87s
# epoch:10, train loss:0.0008, train accuracy:0.928, test accuracy:0.745, time:13.88s
#
# end...
#
#  resnet_50 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0068, train accuracy:0.387, test accuracy:0.483, time:24.17s
# epoch:2, train loss:0.0058, train accuracy:0.502, test accuracy:0.529, time:24.16s
# epoch:3, train loss:0.0055, train accuracy:0.527, test accuracy:0.525, time:24.16s
# epoch:4, train loss:0.0050, train accuracy:0.564, test accuracy:0.528, time:24.16s
# epoch:5, train loss:0.0044, train accuracy:0.611, test accuracy:0.614, time:24.24s
# epoch:6, train loss:0.0051, train accuracy:0.567, test accuracy:0.495, time:24.27s
# epoch:7, train loss:0.0051, train accuracy:0.560, test accuracy:0.547, time:24.27s
# epoch:8, train loss:0.0044, train accuracy:0.617, test accuracy:0.642, time:24.20s
# epoch:9, train loss:0.0037, train accuracy:0.675, test accuracy:0.667, time:24.27s
# epoch:10, train loss:0.0041, train accuracy:0.640, test accuracy:0.651, time:24.35s
#
# end...
#
#  my_resnet_50 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0070, train accuracy:0.383, test accuracy:0.475, time:24.51s
# epoch:2, train loss:0.0053, train accuracy:0.519, test accuracy:0.527, time:24.52s
# epoch:3, train loss:0.0050, train accuracy:0.556, test accuracy:0.522, time:24.50s
# epoch:4, train loss:0.0058, train accuracy:0.491, test accuracy:0.522, time:24.47s
# epoch:5, train loss:0.0048, train accuracy:0.581, test accuracy:0.607, time:24.51s
# epoch:6, train loss:0.0039, train accuracy:0.652, test accuracy:0.638, time:24.50s
# epoch:7, train loss:0.0034, train accuracy:0.702, test accuracy:0.636, time:24.53s
# epoch:8, train loss:0.0031, train accuracy:0.730, test accuracy:0.665, time:24.54s
# epoch:9, train loss:0.0036, train accuracy:0.693, test accuracy:0.649, time:24.52s
# epoch:10, train loss:0.0050, train accuracy:0.565, test accuracy:0.608, time:24.56s
#
# end...
