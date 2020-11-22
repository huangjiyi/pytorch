import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as F


class BCNN(nn.Module):
    """ 实现BiLinear CNN """

    def __init__(self, num_classes=200):
        super(BCNN, self).__init__()
        resnet = model.resnet18(pretrained=False)
        # 去掉最后的池化层和全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.fc = nn.Linear(512**2, num_classes)

    def forward(self, x):
        x = self.features(x)    # N x 512 x h x w
        N, h, w = x.shape[0], x.shape[2], x.shape[3]

        # binear pooling
        x = x.view(N, 512, -1)  # N x 512 x hw
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (h * w)  # N x 512 x 512
        x = x.view(N, -1)   # N x 512*512
        x = torch.sign(x) * torch.sqrt(torch.abs(x))
        x = F.normalize(x)  # N x 512*512

        # classifier
        x = self.fc(x)

        return x


if __name__ == '__main__':
    " test code "
    # net = BCNN(num_classes=200)
    # X = torch.randn(10, 3, 224, 224)
    # print(net(X).shape)
