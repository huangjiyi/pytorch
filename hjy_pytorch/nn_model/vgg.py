import torch
import torch.nn as nn

"""
    VGG：
    （1）backbone可以分为5个卷积块，每个卷积块由若干个卷积核个数（输出通道）相同的卷积层组成；
    （2）所有卷积层均采用3×3的卷积核以及ReLU激活函数
    （3）每经过一个卷积块，都要进行一次最大池化操作，窗口大小为2，步长为2，之后下一个
         卷积块中每层卷积层的卷积核个数增加一倍，直至达到512(最初为64)
    （4）要求输入大小为224*224的图像，经过backbone后得到512*7*7的feature map，然后展开成长向量
         输入到由3个全连接层[4096,4096,classes_num]构成的分类器。
    （5）由于经典VGG网络中参数量庞大，训练时容易出现显存不足，我们VGG网络中设置了一个比例ratio，
         令每个卷积层的卷积核个数和全连接层神经元的个数都除以ratio，令网络变得更窄，减少参数量。
"""


class VGG(nn.Module):
    """ 实现VGG网络 """

    def __init__(self, conv, fc_arch, init_weights=True):
        super(VGG, self).__init__()
        self.conv = conv
        self.fc = nn.Sequential(
            nn.Linear(fc_arch[0], fc_arch[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_arch[1], fc_arch[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_arch[2], fc_arch[3])
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feature = torch.flatten(self.conv(x), 1)
        output = self.fc(feature)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def conv_layers(in_channels, conv_arch, batch_norm, ratio):
    """ 建立backbone """
    layers = []
    for out_channels in conv_arch:
        if out_channels == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels //= ratio
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
    return nn.Sequential(*layers)


# 不同VGG网络backbone卷积层参数字典
conv_archs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


def _vgg(in_channels, classes_num, conv_arch, batch_norm=False, ratio=8, **kwargs):
    """ 返回给定参数的VGG网络 """
    fc_arch = (512 * 7 * 7 // ratio, 4096 // ratio, 4096 // ratio, classes_num)
    model = VGG(conv_layers(in_channels, conv_arch, batch_norm, ratio), fc_arch, **kwargs)
    return model


def vgg_11(in_channels=3, classes_num=10, batch_norm=False, ratio=8, **kwargs):
    """ 返回经典VGG_11网络，可以加批归一化算法，可以变窄以减小参数量 """
    return _vgg(in_channels, classes_num, conv_archs['11'], batch_norm, ratio, **kwargs)


def vgg_16(in_channels=3, classes_num=10, batch_norm=False, ratio=8, **kwargs):
    """ 返回经典VGG_16网络，可以加批归一化算法，可以变窄以减小参数量 """
    return _vgg(in_channels, classes_num, conv_archs['16'], batch_norm, ratio, **kwargs)


""" 在CIFAR10数据集上的测试结果 """
# 参数：learning_rate = 0.001，optim.Adam(default)，CrossEntropyLoss()
#  VGG_11 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0072, train accuracy:0.307, test accuracy:0.443, time:31.55s
# epoch:2, train loss:0.0053, train accuracy:0.510, test accuracy:0.568, time:31.80s
# epoch:3, train loss:0.0043, train accuracy:0.610, test accuracy:0.638, time:31.63s
# epoch:4, train loss:0.0038, train accuracy:0.664, test accuracy:0.661, time:31.83s
# epoch:5, train loss:0.0033, train accuracy:0.706, test accuracy:0.680, time:31.95s
# epoch:6, train loss:0.0029, train accuracy:0.739, test accuracy:0.697, time:31.96s
# epoch:7, train loss:0.0026, train accuracy:0.765, test accuracy:0.704, time:32.02s
# epoch:8, train loss:0.0024, train accuracy:0.788, test accuracy:0.722, time:31.91s
# epoch:9, train loss:0.0022, train accuracy:0.808, test accuracy:0.710, time:32.10s
# epoch:10, train loss:0.0020, train accuracy:0.825, test accuracy:0.720, time:31.98s
#
# end...
#
#  VGG_16 :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0074, train accuracy:0.296, test accuracy:0.415, time:54.81s
# epoch:2, train loss:0.0053, train accuracy:0.512, test accuracy:0.563, time:55.15s
# epoch:3, train loss:0.0043, train accuracy:0.612, test accuracy:0.649, time:54.98s
# epoch:4, train loss:0.0036, train accuracy:0.674, test accuracy:0.675, time:54.70s
# epoch:5, train loss:0.0032, train accuracy:0.717, test accuracy:0.694, time:55.21s
# epoch:6, train loss:0.0028, train accuracy:0.749, test accuracy:0.715, time:56.00s
# epoch:7, train loss:0.0025, train accuracy:0.776, test accuracy:0.725, time:55.88s
# epoch:8, train loss:0.0023, train accuracy:0.798, test accuracy:0.715, time:55.08s
# epoch:9, train loss:0.0021, train accuracy:0.818, test accuracy:0.735, time:54.60s
# epoch:10, train loss:0.0019, train accuracy:0.833, test accuracy:0.738, time:54.67s
#
# end...
#
#  VGG_11_BN :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0059, train accuracy:0.439, test accuracy:0.549, time:38.26s
# epoch:2, train loss:0.0042, train accuracy:0.617, test accuracy:0.645, time:38.20s
# epoch:3, train loss:0.0035, train accuracy:0.686, test accuracy:0.696, time:38.32s
# epoch:4, train loss:0.0031, train accuracy:0.724, test accuracy:0.721, time:38.41s
# epoch:5, train loss:0.0029, train accuracy:0.750, test accuracy:0.736, time:38.15s
# epoch:6, train loss:0.0026, train accuracy:0.769, test accuracy:0.740, time:38.30s
# epoch:7, train loss:0.0024, train accuracy:0.787, test accuracy:0.751, time:38.21s
# epoch:8, train loss:0.0022, train accuracy:0.803, test accuracy:0.763, time:38.42s
# epoch:9, train loss:0.0021, train accuracy:0.818, test accuracy:0.770, time:37.98s
# epoch:10, train loss:0.0019, train accuracy:0.828, test accuracy:0.772, time:38.47s
#
# end...
#
#  VGG_16_BN :
#
# training on  cuda
# start...
#
# epoch:1, train loss:0.0064, train accuracy:0.388, test accuracy:0.524, time:66.91s
# epoch:2, train loss:0.0047, train accuracy:0.573, test accuracy:0.606, time:66.87s
# epoch:3, train loss:0.0039, train accuracy:0.652, test accuracy:0.680, time:66.83s
# epoch:4, train loss:0.0033, train accuracy:0.707, test accuracy:0.710, time:66.60s
# epoch:5, train loss:0.0029, train accuracy:0.743, test accuracy:0.733, time:66.45s
# epoch:6, train loss:0.0026, train accuracy:0.768, test accuracy:0.748, time:66.54s
# epoch:7, train loss:0.0024, train accuracy:0.789, test accuracy:0.758, time:66.61s
# epoch:8, train loss:0.0022, train accuracy:0.810, test accuracy:0.768, time:66.46s
# epoch:9, train loss:0.0020, train accuracy:0.824, test accuracy:0.781, time:66.65s
# epoch:10, train loss:0.0019, train accuracy:0.837, test accuracy:0.781, time:66.44s
