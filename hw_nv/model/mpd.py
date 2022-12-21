import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

CHANNELS = [1, 32, 128, 512, 1024, 1024] #took channels sizes from authors code, its different from ones in the paper

class MPD_subdiscr(nn.Module):
    def __init__(self, p):
        super(MPD_subdiscr, self).__init__()
        self.p = p

        layers = []
        for l in range(5):
            layers.append(weight_norm(nn.Conv2d(
                in_channels=CHANNELS[l],
                out_channels=CHANNELS[l+1],
                kernel_size=(5,1),
                stride=(3,1) if l!=4 else 1,
                padding=(2,0)
            )))
        self.last_conv = weight_norm(nn.Conv2d(
            in_channels=CHANNELS[-1],
            out_channels=1,
            kernel_size=(3,1),
            stride=1,
            padding=(1,0)
        ))
        self.layers = nn.ModuleList(layers) 

    def forward(self, x):
        fmap = []
        bs, ch, t = x.shape
        # print(f'shape {x.shape}')
        x = F.pad(x, (0, (self.p - t % self.p) % self.p), "reflect")
        x = x.view(bs, ch, -1, self.p) 
        # print(f'post shape {x.shape}')
        for module in self.layers:
            x = F.leaky_relu(module(x), 0.1)
            fmap.append(x)
        x = self.last_conv(x)
        fmap.append(x)
        return x, fmap


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MPD, self).__init__()
        self.subdiscriminators = nn.ModuleList([MPD_subdiscr(p) for p in periods])
            
    def forward(self, x):
        fmaps = []
        preds = []
        for subdiscr in self.subdiscriminators:
            pred, fmap = subdiscr(x)
            preds.append(pred)
            fmaps.append(fmap)
        return preds, fmaps