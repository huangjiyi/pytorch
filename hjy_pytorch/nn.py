import torch
import torch.nn as nn
import torch.nn.functional as fun


# class InceptionV3(nn.Module):
#     def __init__(self, in_channels):
#         super(InceptionV3, self).__init__()
#         # 通路1：1×1卷积层 + 2个3×3卷积层
#
#         # 通路2：1×1卷积层 + 3×3卷积层
#
#         # 通路3：最大池化层 + 1×1卷积层
#
#         # 通路4：1×1卷积层
#         self.p1_1 = nn.Conv2d(in_channels, )

class FlattenLayer(nn.Module):
    """展开层：将图像展开为行向量"""

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return fun.avg_pool2d(x, kernel_size=x.size()[2:])


# DenseNet
def conv_block(in_channels, out_channels):
    """将批归一化、激活函数和卷积合成一个结构块"""
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk


class DenseBlock(nn.Module):
    """ 稠密块由多个conv_block组成，并且每个conv_block的输出
        都在后续conv_block输出的通道维上相连 """

    def __init__(self, in_channels, out_channels, num_convs):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(in_channels, out_channels):
    """ 两个Dense块之间用过渡块连接，过渡块主要用于降低模型复杂度
        即使用1×1卷积核降低通道数，和2×2平均池化减半高和宽 """
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )
    return blk


class DenseNet(nn.Module):
    def __init__(self, in_channels, num_output):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels, grow_rate = 64, 32
        num_convs_list = [4, 4, 4, 4]

        for i, num_convs in enumerate(num_convs_list):
            block = DenseBlock(num_channels, grow_rate, num_convs)
            self.net.add_module("DenseBlock_%d" % (i + 1), block)

            num_channels = block.out_channels
            if i != len(num_convs_list) - 1:
                self.net.add_module("transition_block_%d" % (i + 1),
                                    transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net.add_module("BatchNorm", nn.BatchNorm2d(num_channels))
        self.net.add_module("ReLU", nn.ReLU())
        self.net.add_module("GlobalAvgPool", GlobalAvgPool2d())
        self.net.add_module("fc", nn.Sequential(FlattenLayer(),
                                                nn.Linear(num_channels, num_output)))

    def forward(self, X):
        return self.net(X)
