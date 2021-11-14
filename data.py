import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

from ptpt.log import error

import os
import PIL
import glob
import pandas as pd
from functools import partial

class CelebADataset(torch.utils.data.Dataset):
    """CelebA Dataset class"""

    def __init__(self, 
             root,
             split="train",
             target_type="attr",
             transform=None,
             target_transform=None,
         ):
        """
        """

        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        
        split_ = split_map[torchvision.datasets.utils.verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.csv"), delim_whitespace=False, header=0, index_col=0)
        # This file is not available in Kaggle
        # identity = pd.read_csv(fn("identity_CelebA.csv"), delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.csv"), delim_whitespace=False, header=0, index_col=0)

        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        self.filename = splits[mask].index.values
        # self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        # self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, 
                                        "img_align_celeba", 
                                        "img_align_celeba", 
                                        self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f"Target type {t} is not recognized")

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

def get_dataset(name, shape):
    if name in ['cifar', 'cifar10']:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    elif name in ['celeba']:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(shape[1:])
        ])
        train_dataset = CelebADataset('data/celeba', split='train', transform=transform)
        test_dataset = CelebADataset('data/celeba', split='valid', transform=transform)
    else:
        msg = f"Unknown dataset '{name}'!"
        error(name)
        raise ValueError(msg)

    return train_dataset, test_dataset
