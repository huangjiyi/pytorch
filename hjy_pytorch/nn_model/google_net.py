import torch
import torch.nn as nn
import torch.nn.functional as fun
import scipy.stats as stats


class BasicConv2d(nn.Module):
    """ 将卷积、批归一化、ReLU合并成一个模块 """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return fun.relu(x, inplace=True)


class Inception(nn.Module):
    """ 实现经典GoogLeNet中的Inception结构 """

    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3[0], kernel_size=1),
            BasicConv2d(ch3x3[0], ch3x3[1], kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5[0], kernel_size=1),
            BasicConv2d(ch5x5[0], ch5x5[1], kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    """ 经典GoogLeNet网络实现，输入图像尺寸：224*224 """

    def __init__(self, in_channels, classes_num, init_weights=True):
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = Inception(192, 64, [96, 128], [16, 32], 32)
        self.inception_3b = Inception(256, 128, [128, 192], [32, 96], 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = Inception(480, 192, [96, 208], [16, 48], 64)
        self.inception_4b = Inception(512, 160, [112, 224], [24, 64], 64)
        self.inception_4c = Inception(512, 128, [128, 256], [24, 64], 64)
        self.inception_4d = Inception(512, 112, [144, 288], [32, 64], 64)
        self.inception_4e = Inception(528, 256, [160, 320], [32, 128], 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = Inception(832, 256, [160, 320], [32, 128], 128)
        self.inception_5b = Inception(832, 384, [192, 384], [48, 128], 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, classes_num)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)

        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = torch.flatten(self.avgpool(x), 1)

        x = self.fc(self.dropout(x))
        return x

    def _initialize_weights(self):
        """ 权值初始化方法直接copy源码 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def googlenet(in_channels=3, classes_num=10, **kwargs):
    """  返回经典GoogLeNet网络模型 """
    return GoogLeNet(in_channels, classes_num, **kwargs)
