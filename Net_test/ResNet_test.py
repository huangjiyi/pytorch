# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as model
import sys
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '1'
sys.path.append("..")
import hjy_pytorch as hjy

""" 读取数据 """
batch_size = 256
train_iter, test_iter = hjy.load_data_CIFAR10(batch_size)

in_channels, classes_num = 3, 10
""" 定义模型 """
resnet_18 = model.resnet18(num_classes=classes_num)
my_resnet_18 = hjy.resnet18(in_channels, classes_num)
resnet_50 = model.resnet50(num_classes=classes_num)
my_resnet_50 = hjy.resnet50(in_channels, classes_num)

resnets = [resnet_18, my_resnet_18, resnet_50, my_resnet_50]
resnets_names = ["resnet_18", "my_resnet_18", "resnet_50", "my_resnet_50"]

for i in range(len(resnets)):
    print('\n', resnets_names[i], ':')
    net = resnets[i]
    """ 定义优化算法 """
    learning_rate, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """ 开始训练 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hjy.train_classifier(net, nn.CrossEntropyLoss(), optimizer,
                         num_epochs, train_iter, test_iter, device)
