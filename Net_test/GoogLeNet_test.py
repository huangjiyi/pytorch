# -*- coding: utf-8 -*-
import torch
from torch import nn
import sys
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0'
sys.path.append("..")
import hjy_pytorch as hjy


def conv_block(in_channels, out_channels):
    """将批归一化、激活函数和卷积合成一个结构"""
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


# 由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。
# 过渡层用来控制模型复杂度。它通过1×1卷积层来减小通道数，
# 并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )
    return blk


# ############## 建立DenseNet模型 ###################
# DenseNet首先使用同ResNet一样的单卷积层和最大池化层
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 接下来使用4个稠密块，相邻稠密块间使用过渡层
num_channels, grow_rate = 64, 32
num_convs_list = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_list):
    block = DenseBlock(num_channels, grow_rate, num_convs)
    net.add_module("DenseBlock_%d" % (i + 1), block)

    num_channels = block.out_channels
    if i != len(num_convs_list) - 1:
        net.add_module("transition_block_%d" % (i + 1),
                       transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 同ResNet一样，最后接上全局池化层和全连接层来输出
net.add_module("BatchNorm", nn.BatchNorm2d(num_channels))
net.add_module("ReLU", nn.ReLU())
net.add_module("GlobalAvgPool", hjy.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(hjy.FlattenLayer(),
                                   nn.Linear(num_channels, 10))
               )
net = hjy.DenseNet(1, 10)

# ############# 读取数据 ###############
batch_size = 256
train_iter, test_iter = hjy.load_data_fashion_mnist(batch_size, resize=(96, 96))

# ############# 开始训练 ###############
leaning_rate, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=leaning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hjy.train_classifier(net, nn.CrossEntropyLoss(), optimizer, num_epochs, train_iter, test_iter, device)

# ############ 训练结果 ################
# epoch:1, train loss:0.0017, train accuracy:0.845, test accuracy:0.891, time:190.40s
# epoch:2, train loss:0.0010, train accuracy:0.902, test accuracy:0.900, time:204.12s
# epoch:3, train loss:0.0009, train accuracy:0.915, test accuracy:0.910, time:210.16s
# epoch:4, train loss:0.0008, train accuracy:0.925, test accuracy:0.917, time:219.00s
# epoch:5, train loss:0.0007, train accuracy:0.932, test accuracy:0.918, time:218.60s

# 实验室服务器结果（快了10倍，垃圾电脑）
# epoch:1, train loss:0.0018, train accuracy:0.840, test accuracy:0.882, time:21.08s
# epoch:2, train loss:0.0011, train accuracy:0.900, test accuracy:0.898, time:21.08s
# epoch:3, train loss:0.0009, train accuracy:0.915, test accuracy:0.909, time:21.09s
# epoch:4, train loss:0.0008, train accuracy:0.923, test accuracy:0.913, time:21.11s
# epoch:5, train loss:0.0007, train accuracy:0.931, test accuracy:0.919, time:21.11s
