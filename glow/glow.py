import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import math

from .utils import HelperModule
from .actnorm import ActNorm
from .coupling import AffineCoupling, AdditiveCoupling
from .conv import InvConv

class Flow(HelperModule):
    def build(self, 
            nb_channels:        int,
            lu:                 bool = True,
            affine_coupling:    bool = False,
            grad_checkpoint:    bool = True,
        ):
        self.norm = ActNorm(nb_channels)
        self.conv = InvConv(nb_channels, lu)

        Coupling = AffineCoupling if affine_coupling else AdditiveCoupling
        self.coupling = Coupling(nb_channels, grad_checkpoint=grad_checkpoint)

    def forward(self, x):
        x, d1 = self.norm(x)
        x, d2 = self.conv(x)
        x, d3 = self.coupling(x)

        return x, d1 + d2 + d3

    def reverse(self, x):
        x = self.coupling.reverse(x)
        x = self.conv.reverse(x)
        x = self.norm.reverse(x)
        return x

class FlowBlock(HelperModule):
    def build(self,
            nb_channels:        int,
            nb_flows:           int,
            squeeze_rate:       int = 2,
            split:              bool = True,
            lu:                 bool = True,
            affine_coupling:    bool = False,
            grad_checkpoint:    bool = True,
        ):
        self.nb_channels = nb_channels
        self.split = split
        self.squeeze_rate = squeeze_rate
        self.flows = nn.ModuleList([
            Flow(
                nb_channels * squeeze_rate**2, 
                lu=lu, 
                affine_coupling=affine_coupling, 
                grad_checkpoint=grad_checkpoint
            )
            for _ in range(nb_flows)
        ])
        if split:
            self.prior = nn.Conv2d((nb_channels*squeeze_rate**2)//2, nb_channels*squeeze_rate**2, 3, padding=1)
            self.prior.weight.data.zero_()
            self.prior.bias.data.zero_()
            self.scale = nn.Parameter(torch.zeros(1, nb_channels*squeeze_rate**2, 1, 1))
        else:
            self.prior = nn.Conv2d(nb_channels*squeeze_rate**2, 2*nb_channels*squeeze_rate**2, 3, padding=1)
            self.prior.weight.data.zero_()
            self.prior.bias.data.zero_()
            self.scale = nn.Parameter(torch.zeros(1, 2*nb_channels*squeeze_rate**2, 1, 1))

    def _squeeze(self, x):
        N, c, hx, wx = x.shape
        return (
            x.view(N, c, hx // self.squeeze_rate, self.squeeze_rate, wx // self.squeeze_rate, self.squeeze_rate)
            .permute(0, 1, 3, 5, 2, 4)
            .contiguous()
            .view(N, c * self.squeeze_rate**2, hx // self.squeeze_rate, wx // self.squeeze_rate)
        )

    def _unsqueeze(self, x):
        N, c, hx, wx = x.shape
        return (
            x.view(N, c // self.squeeze_rate**2, self.squeeze_rate, self.squeeze_rate, hx, wx)
            .permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .view(N, c // self.squeeze_rate**2, hx * self.squeeze_rate, wx * self.squeeze_rate)
        )

    def _gaussian_log_p(self, x, mean, log_sd):
        return -0.5 * math.log(2 * math.pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

    def _gaussian_sample(self, eps, mean, log_sd):
        return mean + torch.exp(log_sd) * eps

    def forward(self, x):
        # C x H x W
        logdet_sum = 0.
        x = self._squeeze(x)
        # C*sr**2 x H//2 x W//2

        for flow in self.flows:
            x, logdet = flow(x)
            logdet_sum = logdet_sum + logdet
        # C*sr**2 x H//2 x W//2

        if self.split:
            x, z = x.chunk(2, dim=1)
            prior_x = x
            # (C*sr**2)//2 x H//2 x W//2
        else:
            prior_x = torch.zeros_like(x)
            # C*sr**2 x H//2 x W//2

        mean, log_sd = (self.prior(prior_x) * torch.exp(self.scale * 3)).chunk(2, dim=1)
        log_p = self._gaussian_log_p(z if self.split else x, mean, log_sd).view(x.shape[0], -1).sum(-1)
        z = z if self.split else x
        return x, logdet_sum, log_p, z

    def reverse(self, x, eps=None, recon=False):
        if recon:
            x = torch.cat([x, eps], dim=1) if self.split else eps
        else:
            prior_x = x if self.split else torch.zeros_like(x)
            mean, log_sd = (self.prior(prior_x) * torch.exp(self.scale * 3)).chunk(2, dim=1)
            z = self._gaussian_sample(eps, mean, log_sd)
            x = torch.cat([x, z], dim=1) if self.split else z

        for flow in self.flows[::-1]:
            x = flow.reverse(x)

        return self._unsqueeze(x)

class Glow(HelperModule):
    def build(self,
            nb_channels:        int,
            nb_blocks:          int,
            nb_flows:           int,
            squeeze_rate:       int = 2,
            lu:                 bool = True,
            affine_coupling:    bool = False,
            grad_checkpoint:    bool = True,
        ):
        self.nb_blocks = nb_blocks
        self.squeeze_rate = squeeze_rate
        self.blocks = nn.ModuleList([
            FlowBlock(
                nb_channels * ((squeeze_rate**2)//2)**i, nb_flows, 
                squeeze_rate=squeeze_rate, 
                lu=lu,
                grad_checkpoint = grad_checkpoint,
                affine_coupling = affine_coupling,
                split = i < (nb_blocks - 1)
            )
            for i in range(nb_blocks)
        ])

    def forward(self, x):
        log_p_sum = 0.
        logdet_sum = 0.
        zs = []

        for blk in self.blocks:
            x, logdet, log_p, z = blk(x)

            logdet_sum = logdet_sum + logdet # not inplace, so not to break "stuff"
            log_p_sum = log_p_sum + log_p
            zs.append(z)
        return log_p_sum, logdet_sum, zs

    def reverse(self, zs, recon=False):
        x = zs[-1]
        for i, blk in enumerate(self.blocks[::-1]):
            x = blk.reverse(x, zs[-(i+1)], recon=recon)
        return x

    def get_latent_shapes(self, image_shape):
        c, h, w = image_shape
        z_shapes = []
        for i in range(self.nb_blocks):
            h //= self.squeeze_rate
            w //= self.squeeze_rate
            c *= (self.squeeze_rate**2)//2 if i < (self.nb_blocks - 1) else self.squeeze_rate**2
            z_shapes.append((c, h, w))
        return z_shapes

if __name__ == '__main__':
    glow = Glow(
        nb_channels=3, 
        nb_blocks=3, 
        nb_flows=2,
        squeeze_rate=2,
        lu=True,
    )
    x = torch.randn(1, 3, 64, 64)
    _, logdet, zs = glow(x)
    xr = glow.reverse(zs)
    print(x[0,0])
    print(xr[0,0])
