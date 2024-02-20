import torchvision
from icecream import ic

# woman: 98
# whale: 95

def get_classes(dataset, data_path):
    if dataset == 'CIFAR100':
        return torchvision.datasets.CIFAR100(data_path, train=True).classes
    elif dataset == 'CIFAR10':
        return torchvision.datasets.CIFAR10(data_path, train=True).classes

def debug_output(s):
    output_filename = './debug_output.txt'
    with open(output_filename, 'a') as f:
        f.write(s + '\n')

debug_output('---------------------------------------------------------------')