import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm


class MSD_subdiscr(nn.Module):
    """
    Sub-discriminator block of MSD
    """
    def __init__(self, norm):
        super(MSD_subdiscr, self).__init__()
        self.layers = nn.ModuleList([ #just copied authors code here because parameters are different from those in MelGAN paper
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.last_conv = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for module in self.layers:
            x = F.leaky_relu(module(x), 0.1)
            fmap.append(x)
        x = self.last_conv(x)
        fmap.append(x)
        return x, fmap


class MSD(nn.Module):
    """
    Multi-Scale Discriminator
    """
    def __init__(self):
        super(MSD, self).__init__()
        self.subdiscriminators = nn.ModuleList([
            MSD_subdiscr(norm=spectral_norm),
            nn.Sequential(
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
                MSD_subdiscr(norm=weight_norm)
                ),
            nn.Sequential(
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
                MSD_subdiscr(norm=weight_norm)
                )
        ])
            
    def forward(self, x):
        fmaps = []
        preds = []
        for subdiscr in self.subdiscriminators:
            pred, fmap = subdiscr(x)
            preds.append(pred)
            fmaps.append(fmap)
        return preds, fmaps