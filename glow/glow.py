import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule
from .actnorm import ActNorm
from .coupling import AffineCoupling
from .conv import InvConv

class Flow(HelperModule):
    def build(self, 
            nb_channels: int,
            lu: bool = True,
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

class Glow(HelperModule):
    def build(self):
        pass

    def forward(self):
        pass

    def reverse(self):
        pass

if __name__ == '__main__':
    flow = Flow(8, lu=False)
    x = torch.randn(4, 8, 16, 16)
    y, logdet = flow(x)
    xr = flow.reverse(y)
    print(x[0,0])
    print(xr[0,0])
