import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule
from .actnorm import ActNorm
from .coupling import AdditiveCoupling, AffineCoupling
from .conv import InvConv

class Flow(HelperModule):
    def build(self):
        pass

    def forward(self):
        pass

    def reverse(self):
        pass

class Glow(HelperModule):
    def build(self):
        pass

    def forward(self):
        pass

    def reverse(self):
        pass
