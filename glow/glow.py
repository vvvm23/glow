import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule
from .actnorm import ActNorm
from .coupling import AffineCoupling
from .conv import InvConv

class Flow(HelperModule):
    def build(self, 
            nb_channels:        int,
            lu:                 bool = True,
        ):
        self.norm = ActNorm(nb_channels)
        self.conv = InvConv(nb_channels, lu)
        self.coupling = AffineCoupling(nb_channels)

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
        ):
        pass

    def forward(self, x):
        pass

    def reverse(self, x, eps=None, recon=False):
        pass

class Glow(HelperModule):
    def build(self,
            nb_channels:        int,
            nb_blocks:          int,
            nb_flows:           int,
            squeeze_rate:       int = 2,
            lu:                 bool = True,
        ):
        self.blocks = nn.ModuleList([
            FlowBlock(
                nb_channels*(2**i), nb_flows, 
                squeeze_rate=squeeze_rate, 
                lu=lu, 
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

    def reverse(self, zs, recon=False):
        x = zs[-1]
        for i, blk in enumerate(self.blocks[::-1]):
            x = blk.reverse(x, z_list[-(i+1)], recon=recon)
        return x

if __name__ == '__main__':
    flow = Flow(8, lu=False)
    x = torch.randn(4, 8, 16, 16)
    y, logdet = flow(x)
    xr = flow.reverse(y)
    print(x[0,0])
    print(xr[0,0])
