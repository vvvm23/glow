import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class AffineCoupling(HelperModule):
    def build(self,
            nb_channels: int,
            hidden_channels: int = 512,
            grad_checkpoint: bool = True,
        ):
        self.grad_checkpoint = grad_checkpoint
        self.net = nn.Sequential(
            nn.Conv2d(nb_channels // 2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, nb_channels, 3, padding=1)
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.net[4].weight.data.zero_()
        self.net[4].bias.data.zero_()

        self.scale = nn.Parameter(torch.zeros(1, nb_channels, 1, 1))

    def forward(self, x):
        N, *_ = x.shape
        x_a, x_b = x.chunk(2, dim=1)
        
        # nn_out = self.net(x_a)
        nn_out = torch.utils.checkpoint.checkpoint(self.net, x_a) if self.grad_checkpoint else self.net(x_a)
        log_s, t = (nn_out * torch.exp(self.scale * 3)).chunk(2, dim=1) # why *3?
        s = torch.sigmoid(log_s + 2) # again, why +2?
        y = (x_b + t) * s

        # return torch.cat([x_a, y], dim=1), torch.sum(torch.log(s).view(N, -1), dim=-1)
        return torch.cat([x_a, y], dim=1), torch.log(s).view(N, -1).sum(dim=-1)

    def reverse(self, x):
        N, *_ = x.shape
        x_a, x_b = x.chunk(2, dim=1)
        
        nn_out = self.net(x_a)
        log_s, t = (nn_out * torch.exp(self.scale * 3)).chunk(2, dim=1) # why *3?
        s = torch.sigmoid(log_s + 2) # again, why +2?
        y = x_b / s - t
        return torch.cat([x_a, y], dim=1)

class AdditiveCoupling(HelperModule):
    def build(self,
            nb_channels: int,
            hidden_channels: int = 512,
            grad_checkpoint: bool = True,
        ):
        self.grad_checkpoint = grad_checkpoint
        self.net = nn.Sequential(
            nn.Conv2d(nb_channels // 2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, nb_channels // 2, 3, padding=1)
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.net[4].weight.data.zero_()
        self.net[4].bias.data.zero_()

        self.scale = nn.Parameter(torch.zeros(1, nb_channels // 2, 1, 1))

    def forward(self, x):
        N, *_ = x.shape
        x_a, x_b = x.chunk(2, dim=1)
        
        nn_out = torch.utils.checkpoint.checkpoint(self.net, x_a) if self.grad_checkpoint else self.net(x_a)
        nn_out = (nn_out * torch.exp(self.scale * 3))
        y = x_b + nn_out

        return torch.cat([x_a, y], dim=1), 0.0

    def reverse(self, x):
        N, *_ = x.shape
        x_a, x_b = x.chunk(2, dim=1)
        
        nn_out = self.net(x_a)
        nn_out = (nn_out * torch.exp(self.scale * 3))
        y = x_b - nn_out

        return torch.cat([x_a, y], dim=1)

if __name__ == '__main__':
    affn = AdditiveCoupling(8)
    x = torch.randn(4, 8, 16, 16)
    y, logdet = affn(x)
    xr = affn.reverse(y)
    print(x[0,0])
    print(xr[0,0])
