from utils.transforms import get_transforms
from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as T
from torchvision.datasets import ImageNet

def get_dataset(name, path, transform, train=True):
    if transform == 'train':
        trans = get_transforms('supervised', name)
    elif transform == 'val':
        trans = get_transforms('eval', name)
    else:
        raise NotImplementedError

    if name == 'CIFAR100':
        return CIFAR100(path, train, trans)
    elif name == 'CIFAR10':
        return CIFAR10(path, train, trans)
    else:
        return ImageNet(path, split='train' if train else 'val', transform=trans)

