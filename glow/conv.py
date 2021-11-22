import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class InvConv(HelperModule):
    def build(self,
            nb_channels: int,
            lu: bool = True,
        ):
        self.lu = lu

        if self.lu:
            weight = torch.randn(nb_channels, nb_channels)
            q = torch.linalg.qr(weight)[0]
            w_p, w_l, w_u = torch.lu_unpack(*q.lu()) 
            w_s = torch.diag(w_u)
            w_u = torch.triu(w_u, 1)
            u_mask = torch.triu(torch.ones_like(w_u), 1)
            l_mask = u_mask.T

            self.register_buffer('w_p', w_p)
            self.register_buffer('u_mask', u_mask)
            self.register_buffer('l_mask', l_mask)
            self.register_buffer('s_sign', torch.sign(w_s))
            self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

            self.w_l = nn.Parameter(w_l)
            # self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
            self.w_s = nn.Parameter(w_s.abs().log())
            self.w_u = nn.Parameter(w_u)
        else:
            # initialize as a random rotation matrix
            self.register_parameter(
                'weight', 
                nn.Parameter(torch.linalg.qr(torch.randn(nb_channels, nb_channels))[0][..., None, None]) 
            )

    def _calc_lu_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight[..., None, None]

    def forward(self, x):
        _, _, hx, wx = x.shape

        weight = self._calc_lu_weight() if self.lu else self.weight
        y = F.conv2d(x, weight)

        if self.lu:
            logdet = hx * wx * torch.sum(self.w_s)
        else:
            # TODO: make typing more flexible (if possible)
            logdet = hx * wx * torch.slogdet(weight.squeeze().double())[1].float()

        return y, logdet

    def reverse(self, x):
        weight = self._calc_lu_weight() if self.lu else self.weight
        return F.conv2d(x, weight.squeeze().inverse()[..., None, None])

if __name__ == '__main__':
    conv = InvConv(8, lu=True)
    x = torch.randn(4,8,16,16)
    y, logdet = conv(x)
    xr = conv.reverse(y)
    
    print(x[0,0])
    print(xr[0,0])
