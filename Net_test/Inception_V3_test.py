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
train_iter, test_iter = hjy.load_data_CIFAR10(batch_size, resize=(149, 149))

in_channels, classes_num = 3, 10

""" 定义模型 """
inception_v3 = model.inception_v3(num_classes=10, aux_logits=False, init_weights=True)
# inception_v3_with_aux = model.inception_v3(num_classes=10, aux_logits=True, init_weights=True)
# my_inception_v3 = hjy.inception_v3(in_channels, classes_num, aux_logits=False)
# my_inception_v3_with_aux = hjy.inception_v3(in_channels, classes_num, aux_logits=True)
#
# inception_v3s = [inception_v3, my_inception_v3, inception_v3_with_aux, my_inception_v3_with_aux]
# inception_v3s_names = ["inception_v3(Pytorch)", "inception_v3(my)",
#                        "inception_v3_with_aux(Pytorch)", "inception_v3_with_aux(my)"]
#
# for i in range(len(inception_v3s)):
#     print('\n', inception_v3s_names[i], ':')
net = inception_v3
""" 定义优化算法 """
learning_rate, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

""" 开始训练 """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hjy.train_inception_net(net, nn.CrossEntropyLoss(), optimizer,
                        num_epochs, train_iter, test_iter, device)

