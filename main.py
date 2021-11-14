#!/usr/bin/env python

import argparse

import torch
import torch.nn.functional as F

from torchvision.utils import save_image

import toml
from math import log, sqrt
from types import SimpleNamespace

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.callbacks import CallbackType
from ptpt.log import debug, info, warning, error, critical
from ptpt.utils import set_seed, get_parameter_count

from glow import Glow
from data import get_dataset

def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    train_dataset, test_dataset = get_dataset(cfg.data['name'], cfg.data['shape'])

    net = Glow(**cfg.glow, grad_checkpoint=not args.no_grad_checkpoint)

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

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
        nb_workers = args.nb_workers,
        save_outputs = not args.no_save,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
        metric_names = ['log_p', 'logdet'],
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        cfg = trainer_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    z_shapes = net.get_latent_shapes(tuple(cfg.data['shape']))
    z_sample = [0.7*torch.randn(args.nb_samples, *zs).to(trainer.device) for zs in z_shapes]

    @torch.inference_mode()
    def callback_sample(trainer):
        debug("saving Glow sample")
        net.eval()
        sample = net.reverse(z_sample).cpu()
        save_image(
            sample,
            trainer.directories['root'] / f"sample-{str(trainer.nb_updates).zfill(6)}.jpg",
            normalize=True,
            nrow=int(sqrt(args.nb_samples)),
            value_range=(-1.0, 1.0),
        )
        net.train()

    trainer.register_callback(CallbackType.EvalEpoch, callback_sample)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/cifar10.toml')
    parser.add_argument('--nb-samples', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-grad-checkpoint', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=4)
    args = parser.parse_args()

    main(args)
