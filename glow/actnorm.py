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

        # TODO: do init function
        
    # TODO: do I need to return logdet?
    def forward(self, x):
        _, _, hx, wx = x.shape
        return self.scale * (x + self.loc), hw * wx * self.scale.abs().log().sum()

    def reverse(self, x):
        return x / self.scale - self.loc
