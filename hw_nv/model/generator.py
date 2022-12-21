import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

class ResBlock(nn.Module): 
    #authors use weight_norm in their implementation, so do i
    def __init__(self, channels, d_r, k_r):
        super(ResBlock, self).__init__()
        self.d_r = d_r
        layers = []
        for m in range(len(d_r)):
            for l in range(len(d_r[m])):
                layers.append(
                    weight_norm(nn.Conv1d(
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=k_r,
                            dilation=d_r[m][l],
                            padding='same'
                        )))
        self.resblock = nn.ModuleList(layers)

    def forward(self, x):
        i = 0
        for m in range(len(self.d_r)):
            res = x
            for l in range(len(self.d_r[m])):
                x = F.leaky_relu(x, 0.1)
                x = self.resblock[i](x)
                i += 1
            x += res
        return x

    def remove_weight_norm(self):
        for module in self.resblock:
            remove_weight_norm(module)


class MRF(nn.Module):
    def __init__(self, channels, D_r, K_r):
        super(MRF, self).__init__()
        self.resblocks = nn.ModuleList([ResBlock(channels, d_r, k_r) for d_r, k_r in zip(D_r, K_r)])


    def forward(self, x):
        outputs = [block(x) for block in self.resblocks]
        return torch.mean(torch.stack(outputs), axis=0) #authors use mean instead of sum in their implementation

    def remove_weight_norm(self):
        for block in self.resblocks:
            block.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, D_r, K_r, k_u, h_u):
        super(Generator, self).__init__()
        self.conv_in = weight_norm(nn.Conv1d(80, h_u, 7, 1, padding=3))
            # in_channels=80,
            # out_channels=h_u,
            # kernel_size=7,
            # dilation=1,
            # padding='same'
            # ))

        layers = []
        for l in range(len(k_u)):
            out_channels = h_u // (2 ** (l+1))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(weight_norm(nn.ConvTranspose1d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=(k_u[l],1),
                stride=k_u[l]/2
            )))
            layers.append(MRF(out_channels, D_r, K_r))
        self.blocks = nn.Sequential(layers)

        self.conv_out = weight_norm(nn.Conv1d(
                in_channels=out_channels,
                out_channels=1,
                kernel_size=(7,1),
                padding='same'
                ))


    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = F.tanh(self.conv_out(F.leaky_relu(x, 0.1)))
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_in)
        for module in self.blocks:
            if isinstance(module, nn.ConvTranspose1d):
                remove_weight_norm(module)
            elif isinstance(module, MRF):
                module.remove_weight_norm()
        remove_weight_norm(self.conv_out)