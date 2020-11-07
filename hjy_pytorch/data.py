import torch.utils.data as data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import sys

sys.path.append("..")


def load_data_fashion_mnist(batch_size, resize=None, root='../data/FashionMNIST'):
    """获取fashion mnist数据集的小批量训练/测试样本生成器"""
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,
                                                   transform=transform)
    
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据，windows上不能用
    else:
        num_workers = 4

    train_iter = data.DataLoader(mnist_train, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def load_data_CIFAR10(batch_size, resize=None, root='../data//CIFAR10'):
    """获取FashionMNIST数据集"""
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    train_data = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                              transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                             transform=transform)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:  # linux系统上可以用额外的进程来加速读取数据
        num_workers = 4

    train_iter = data.DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(test_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
