import torch
import torch.nn as nn
from .google_net import BasicConv2d


def inception_v3(in_channels, classes_num, aux_logits=False, **kwargs):
    """ 返回inception-V3网络模型 """
    return InceptionV3(in_channels, classes_num, aux_logits, **kwargs)


class InceptionV3(nn.Module):
    """ 实现Inception-V3网络 """

    def __init__(self, in_channels, classes_num, aux_logits=False, init_weights=True):
        super(InceptionV3, self).__init__()

        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = BasicConv2d(64, 80, kernel_size=1)
        self.conv5 = BasicConv2d(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception_a1 = InceptionA(192, 32)
        self.inception_a2 = InceptionA(256, 64)
        self.inception_a3 = InceptionA(288, 64)
        self.inception_b = InceptionB(288)
        self.inception_c1 = InceptionC(768, 128)
        self.inception_c2 = InceptionC(768, 160)
        self.inception_c3 = InceptionC(768, 160)
        self.inception_c4 = InceptionC(768, 192)
        if aux_logits:
            self.aux = InceptionAux(768, classes_num)
        else:
            self.aux = None

        self.inception_d = InceptionD(768)
        self.inception_e1 = InceptionE(1280)
        self.inception_e2 = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, classes_num)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool1(self.conv3(self.conv2(self.conv1(x))))
        x = self.maxpool2(self.conv5(self.conv4(x)))
        x = self.inception_a3(self.inception_a2(self.inception_a1(x)))
        x = self.inception_b(x)
        x = self.inception_c4(self.inception_c3(self.inception_c2(self.inception_c1(x))))

        if self.aux is not None and self.training:
            aux_out = self.aux(x)
        else:
            aux_out = None

        x = self.inception_d(x)
        x = self.inception_e2(self.inception_e1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(self.dropout(x))

        if self.aux_logits and self.training:
            return out, aux_out
        else:
            return out


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_ch):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, out_channels=64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=48, kernel_size=1),
            BasicConv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        )

        self.branch3x3x2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=64, kernel_size=1),
            BasicConv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            BasicConv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, out_channels=pool_ch, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3x2 = self.branch3x3x2(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3x2, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels, out_channels=384, kernel_size=3, stride=2)
        self.branch3x3x2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=64, kernel_size=1),
            BasicConv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            BasicConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3x2 = self.branch3x3x2(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch3x3x2, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, ch_7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, out_channels=192, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=ch_7x7, kernel_size=1),
            BasicConv2d(in_channels=ch_7x7, out_channels=ch_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels=ch_7x7, out_channels=192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7x2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=ch_7x7, kernel_size=1),
            BasicConv2d(in_channels=ch_7x7, out_channels=ch_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels=ch_7x7, out_channels=ch_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels=ch_7x7, out_channels=ch_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels=ch_7x7, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, out_channels=192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7x2 = self.branch7x7x2(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7x2, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=192, kernel_size=1),
            BasicConv2d(in_channels=192, out_channels=320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, out_channels=192, kernel_size=1),
            BasicConv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch7x7, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2_1 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2_2 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3db_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3db_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3db_2_3_1 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3db_2_3_2 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2_1 = self.branch3x3_2_1(branch3x3_1)
        branch3x3_2_2 = self.branch3x3_2_2(branch3x3_1)

        branch3x3db_1 = self.branch3x3db_1(x)
        branch3x3db_2 = self.branch3x3db_2(branch3x3db_1)
        branch3x3db_2_3_1 = self.branch3x3db_2_3_1(branch3x3db_2)
        branch3x3db_2_3_2 = self.branch3x3db_2_3_2(branch3x3db_2)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3_2_1, branch3x3_2_2,
                   branch3x3db_2_3_1, branch3x3db_2_3_2, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """ 使用Pytorch源码给出的结构 """

    def __init__(self, in_channels, classes_num):
        super(InceptionAux, self).__init__()

        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels, out_channels=128, kernel_size=1)
        self.conv2 = BasicConv2d(128, 768, kernel_size=5)
        self.conv2_t = BasicConv2d(128, 768, kernel_size=1)
        self.conv2.stddev = 0.01
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, classes_num)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = self.avgpool1(x)
        x = self.conv1(x)
        if x.shape[2] >=5 and x.shape[3] >=5:
            x = self.conv2(x)
        else:
            x = self.conv2_t(x)
        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
