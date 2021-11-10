import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

# TODO: add LU version
class InvConv(HelperModule):
    def build(self,
            nb_channels: int,
            lu: bool = True,
        ):
        self.lu = lu

        if self.lu:
            raise NotImplementedError("LU decomposition method not implemented yet!")
        else:
            self.register_parameter(
                'weight', 
                nn.Parameter(torch.linalg.qr(torch.randn(nb_channels, nb_channels))[0][..., None, None]) 
            )

    def _calc_lu_weight(self):
        pass

    def forward(self, x):
        _, _, hx, wx = x.shape

        weight = self._calc_lu_weight() if self.lu else self.weight
        y = F.conv2d(x, weight)

        if self.lu:
            pass
        else:
            # TODO: make typing more flexible (if possible)
            logdet = hx * wx * torch.slogdet(weight.squeeze().double())[1].float()

        return y, logdet

    def reverse(self, x):
        return F.conv2d(x, self.weight.squeeze().inverse()[..., None, None])

if __name__ == '__main__':
    conv = InvConv(8, lu=True)
    x = torch.randn(4,8,16,16)
    y, logdet = conv(x)

    print(y.shape)
    print(logdet)
