import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
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


class CUB200(data.Dataset):
    """ Caltech-UCSD Birds-200-2011数据集 """

    def __init__(self, root, train=True, transform=None):
        super(CUB200, self).__init__()
        self.root = root
        self.train = train
        self.transform_ = transform
        self.classes_file = os.path.join(root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(root, 'train_test_split.txt')  # <image_id> <is_training_image>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        if self.transform_ is not None:
            img = self.transform_(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)
