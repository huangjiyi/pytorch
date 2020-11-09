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
google_net = model.googlenet(num_classes=10, aux_logits=False, init_weights=True)
google_net_with_aux = model.googlenet(num_classes=10, aux_logits=True, init_weights=True)
my_google_net = hjy.google_net(in_channels, classes_num, aux_logits=False)
my_google_net_with_aux = hjy.google_net(in_channels, classes_num, aux_logits=True)

google_nets = [google_net, my_google_net, google_net_with_aux, my_google_net_with_aux]
google_nets_names = ["google_net", "my_google_net",
                     "google_net_with_aux", "my_google_net_with_aux"]

for i in range(len(google_nets)):
    print('\n', google_nets_names[i], ':')
    net = google_nets[i]
    """ 定义优化算法 """
    learning_rate, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """ 开始训练 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hjy.train_inception_net(net, nn.CrossEntropyLoss(), optimizer,
                            num_epochs, train_iter, test_iter, device)
