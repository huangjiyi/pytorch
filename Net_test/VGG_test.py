# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import sys
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '1'
sys.path.append("..")
import hjy_pytorch as hjy

""" 读取数据 """
batch_size = 256
train_iter, test_iter = hjy.load_data_CIFAR10(batch_size, resize=(224, 224))

""" 定义模型 """
in_channels, classes_num = 3, 10
VGG_11 = hjy.vgg_11(in_channels, classes_num)
VGG_16 = hjy.vgg_16(in_channels, classes_num)
VGG_11_BN = hjy.vgg_11(in_channels, classes_num, batch_norm=True)
VGG_16_BN = hjy.vgg_16(in_channels, classes_num, batch_norm=True)

VGG_nets = [VGG_11, VGG_16, VGG_11_BN, VGG_16_BN]
VGG_nets_names = ["VGG_11", "VGG_16", "VGG_11_BN", "VGG_16_BN"]

for i in range(len(VGG_nets)):
    print('\n', VGG_nets_names[i], ':')
    net = VGG_nets[i]
    """ 定义优化算法 """
    learning_rate, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """ 开始训练 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hjy.train_classifier(net, nn.CrossEntropyLoss(), optimizer,
                         num_epochs, train_iter, test_iter, device)
