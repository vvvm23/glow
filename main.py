#!/usr/bin/env python
import logging
logger = logging.getLogger('matplotlib.font_manager')
logger.setLevel(logging.CRITICAL)
logger.disabled = True

logger = logging.getLogger('PIL.PngImagePlugin')
logger.setLevel(logging.CRITICAL)
logger.disabled = True

import argparse

import torch
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid

import numpy as np
import matplotlib.pyplot as plt

import toml
from math import log, sqrt
from types import SimpleNamespace
from pathlib import Path

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.callbacks import CallbackType
from ptpt.log import debug, info, warning, error, critical
from ptpt.utils import set_seed, get_parameter_count, get_device

from glow import Glow
from data import get_dataset

@torch.inference_mode()
def interpolation_test(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    train_dataset, test_dataset = get_dataset(**cfg.data)
    idx = np.random.choice(len(test_dataset), 2*args.nb_samples, replace=False)
    batch_a = torch.stack([test_dataset.__getitem__(i)[0] for i in idx[:args.nb_samples]]).to(device)
    batch_b = torch.stack([test_dataset.__getitem__(i)[0] for i in idx[args.nb_samples:]]).to(device)

    device = get_device(not args.no_cuda)
    net = Glow(**cfg.glow, grad_checkpoint=not args.no_grad_checkpoint)

    @torch.no_grad()
    def dry_run(): 
        idx = np.random.choice(len(train_dataset), cfg.trainer['batch_size'], replace=False)
        X = torch.stack([train_dataset.__getitem__(i)[0] for i in idx], dim=0)
        net(X + torch.randn_like(X) / (2**cfg.data['nb_bits']))
    dry_run()

    if args.resume:
        chk = torch.load(args.resume, map_location=device)
        net.load_state_dict(chk['net'])
    net = net.to(device)
    net.eval()

    _, _, latent_a = net(batch_a)
    _, _, latent_b = net(batch_b)

    latent_a = [args.temperature * l for l in latent_a]
    latent_b = [args.temperature * l for l in latent_b]

    save_dir = Path('samples-interpolate/')
    save_dir.mkdir(exist_ok=True)

    for i, a in enumerate(torch.cat([torch.linspace(0.00, 1.00, 100), torch.linspace(1.00, 0.00, 100)])):
        sample = net.reverse([a * la + (1.0 - a) * lb for la, lb in zip(latent_a, latent_b)]).cpu()
        save_image(
            sample,
            save_dir / f"interpolation-{str(i).zfill(4)}.jpg",
            nrow=int(sqrt(args.nb_samples)),
        )

@torch.inference_mode()
def temperature_test(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    train_dataset, test_dataset = get_dataset(**cfg.data)

    device = get_device(not args.no_cuda)
    net = Glow(**cfg.glow, grad_checkpoint=not args.no_grad_checkpoint)

    @torch.no_grad()
    def dry_run(): 
        idx = np.random.choice(len(train_dataset), cfg.trainer['batch_size'], replace=False)
        X = torch.stack([train_dataset.__getitem__(i)[0] for i in idx], dim=0)
        net(X + torch.randn_like(X) / (2**cfg.data['nb_bits']))
    dry_run()

    if args.resume:
        chk = torch.load(args.resume, map_location=device)
        net.load_state_dict(chk['net'])
    net = net.to(device)
    net.eval()

    z_shapes = net.get_latent_shapes(cfg.data['shape'])
    z_sample = [torch.randn(args.nb_samples, *zs).to(device) for zs in z_shapes]

    save_dir = Path('samples-vtemp/')
    save_dir.mkdir(exist_ok=True)
    
    for i, t in enumerate(torch.cat([torch.linspace(0.0, 2.0, 40), torch.linspace(2.0, 0.0, 40)])):
        sample = net.reverse([t*z for z in z_sample]).cpu()
        save_image(
            sample,
            save_dir / f"temperature-vary-{str(i).zfill(4)}.jpg",
            nrow=int(sqrt(args.nb_samples)),
        )
    # sample = make_grid(sample, nrow=int(sqrt(args.nb_samples)))
    # plt.imshow(sample.permute(1,2,0))
    # plt.show()

@torch.inference_mode()
def sample(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    train_dataset, test_dataset = get_dataset(**cfg.data)

    device = get_device(not args.no_cuda)
    net = Glow(**cfg.glow, grad_checkpoint=not args.no_grad_checkpoint)

    @torch.no_grad()
    def dry_run(): 
        idx = np.random.choice(len(train_dataset), cfg.trainer['batch_size'], replace=False)
        X = torch.stack([train_dataset.__getitem__(i)[0] for i in idx], dim=0)
        net(X + torch.randn_like(X) / (2**cfg.data['nb_bits']))
    dry_run()

    if args.resume:
        chk = torch.load(args.resume, map_location=device)
        net.load_state_dict(chk['net'])
    net = net.to(device)
    net.eval()

    z_shapes = net.get_latent_shapes(cfg.data['shape'])
    z_sample = [args.temperature*torch.randn(args.nb_samples, *zs).to(device) for zs in z_shapes]

    sample = net.reverse(z_sample).cpu()
    save_image(
        sample,
        "sample.jpg",
        nrow=int(sqrt(args.nb_samples)),
    )
    sample = make_grid(sample, nrow=int(sqrt(args.nb_samples)))
    plt.imshow(sample.permute(1,2,0))
    plt.show()

def main(args):
    if args.sample:
        if args.sample_mode == 'normal':
            sample(args)
        elif args.sample_mode == 'vtemp':
            temperature_test(args)
        elif args.sample_mode == 'interpolate':
            interpolation_test(args)
        else:
            msg = f"Unrecognized sample mode '{args.sample_mode}!'"
            error(msg)
            raise ValueError(msg)
        exit()

    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)

    train_dataset, test_dataset = get_dataset(**cfg.data)
    nb_pixels = torch.numel(test_dataset.__getitem__(0)[0])

    net = Glow(**cfg.glow, grad_checkpoint=not args.no_grad_checkpoint)

    @torch.no_grad()
    def dry_run(): 
        idx = np.random.choice(len(train_dataset), cfg.trainer['batch_size'], replace=False)
        X = torch.stack([train_dataset.__getitem__(i)[0] for i in idx], dim=0)
        net(X + torch.randn_like(X) / (2**cfg.data['nb_bits']))
    dry_run()

    def calc_loss(log_p, logdet, nb_pixels):
        loss = -log(2**cfg.data['nb_bits']) * nb_pixels
        loss = loss + logdet + log_p

        return (
            (-loss / (log(2) * nb_pixels)).mean(),
            (log_p / (log(2) * nb_pixels)).mean(),
            (logdet / (log(2) * nb_pixels)).mean(),
        )

    def loss_fn(net, batch):
        # X = batch[0] * 2.0 - 1.0
        X, _ = batch

        log_p, logdet, _ = net(X + torch.rand_like(X) / (2**cfg.data['nb_bits'])) 
        logdet = logdet.mean()

        return calc_loss(log_p, logdet, nb_pixels)

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
    
    z_shapes = net.get_latent_shapes(test_dataset.__getitem__(0)[0].shape)
    z_sample = [args.temperature*torch.randn(args.nb_samples, *zs).to(trainer.device) for zs in z_shapes]

    @torch.inference_mode()
    def callback_sample(trainer):
        if not trainer.cfg.save_outputs:
            return

        debug("saving Glow sample")
        net.eval()
        sample = net.reverse(z_sample).cpu()
        save_image(
            sample,
            trainer.directories['root'] / f"sample-{str(trainer.nb_updates).zfill(6)}.jpg",
            nrow=int(sqrt(args.nb_samples)),
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
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--sample-mode', type=str, default='normal', choices=['normal', 'vtemp', 'interpolate'])
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    main(args)
