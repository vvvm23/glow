from torchvision import datasets
from torchvision import transforms

from ptpt.log import error

def get_dataset(name):
    if name in ['cifar', 'cifar10']:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    else:
        msg = f"Unknown dataset '{name}'!"
        error(name)
        raise ValueError(msg)

    return train_dataset, test_dataset
