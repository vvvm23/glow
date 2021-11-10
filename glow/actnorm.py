import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class ActNorm(HelperModule):
    def build(self, 
            nb_channels: int, 
        ):
        self.loc = nn.Parameter(torch.zeros(1, nb_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, nb_channels, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def intialize(self, x):
        _, c, _, _ = x.shape
        flatten = x.permute(1, 0, 2, 3).contiguous().view(c, -1)

        mean = flatten.mean(-1)[..., None, None, None].permute(1, 0, 2, 3)
        std = flatten.std(-1)[..., None, None, None].permute(1, 0, 2, 3)

        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

        self.initialized.fill_(1)
        
    def forward(self, x):
        if not self.initialized.item():
            self.intialize(x)

        _, _, hx, wx = x.shape
        return self.scale * (x + self.loc), hx * wx * self.scale.abs().log().sum()

    def reverse(self, x):
        return x / self.scale - self.loc

if __name__ == '__main__':
    norm = ActNorm(8)
    x = torch.randn(4, 8, 16, 16)
    y, logdet = norm(x)
    xr = norm.reverse(y)
    print(x[0,0])
    print(xr[0,0])
