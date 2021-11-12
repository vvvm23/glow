#!/usr/bin/env python

import argparse

import torch
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

from math import log

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.utils import set_seed, get_parameter_count

from glow import Glow

def main(args):
    seed = set_seed(None)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    net = Glow(
        nb_channels = 1,
        nb_blocks = 2,
        nb_flows = 5,
    )

    def loss_fn(net, batch):
        X, _ = batch
        _, c, h, w = X.shape
        nb_pixels = c*h*w

        log_p, logdet, _ = net(X + torch.rand_like(X) / 256) # TODO: remove hardcoded quant levels
        logdet = logdet.mean()
        
        loss = log(256) * nb_pixels - logdet - log_p
        loss = loss / (log(2) * nb_pixels)
        loss = loss.mean()

        log_p = log_p / (log(2) * nb_pixels)
        log_p = log_p.mean()

        logdet = logdet / (log(2) * nb_pixels)
        logdet = logdet.mean()

        return loss, log_p, logdet

    cfg = TrainerConfig(
        exp_name = 'mnist-glow',
        batch_size = 1024,
        learning_rate = 4e-4,
        nb_workers = 8,
        save_outputs = False,
        metric_names = ['log_p', 'logdet']
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        cfg = cfg,
    )

    trainer.train()

if __name__ == '__main__':
    print("Aloha, World!")
    main(None)
