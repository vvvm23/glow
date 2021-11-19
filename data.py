import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

import numpy as np

from ptpt.log import error

import os
import PIL
import glob
import pandas as pd
from functools import partial
from typing import Tuple

def get_dataset(
        name:               str, 
        path:               str = None, 
        eval_path:          str = None, 
        eval_percentage:    float = 0.1, 
        shape: Tuple[int, int, int] = None,
    ):
    if name in ['image', 'custom']:
        transform=transforms.Compose([
            transforms.Resize(shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = datasets.ImageFolder(path, transform=transform)
        if eval_path:
            train_dataset = datset
            test_dataset = datasets.ImageFolder(eval_path, transform=transform)
        else:
            idx = np.arange(len(dataset))
            nb_eval = int(len(dataset) * eval_percentage)

            eval_idx = np.random.choice(idx, (nb_eval,), replace=False)

            mask = np.ones_like(idx)
            mask[eval_idx] = 0

            train_idx = idx[mask == 1]
            
            train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
            test_dataset = torch.utils.data.Subset(dataset, eval_idx.tolist())

    elif name in ['cifar', 'cifar10']:
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
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

if __name__ == '__main__':
    from torchvision.utils import save_image
    _, test_dataset = get_dataset('image', '~/datasets/flowers/', shape=(3, 64, 64))
    save_image(
        torch.stack([test_dataset.__getitem__(i)[0] for i in range(16)], dim=0),
        'test.jpg',
        normalize=True,
        nrow=4,
        value_range=(-1.0, 1.0),
    )
