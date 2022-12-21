from torch import nn
from hw_nv.model.mpd import MPD
from hw_nv.model.msd import MSD


class Discriminator(nn.Module):
    def __init__(self, periods):
        super(Discriminator, self).__init__()
        self.mpd = MPD(periods)
        self.msd = MSD()


    def forward(self, y_true, y_false):
        print(f'true shape {y_true.shape}, false shape {y_false.shape}')
        mpd_f_preds, mpd_f_fmaps = self.mpd(y_false)
        mpd_t_preds, mpd_t_fmaps = self.mpd(y_true)
        msd_f_preds, msd_f_fmaps = self.msd(y_false)
        msd_t_preds, msd_t_fmaps = self.msd(y_true)
        return mpd_f_preds, msd_f_preds, mpd_t_preds, msd_t_preds, \
            mpd_f_fmaps, msd_f_fmaps, mpd_t_fmaps, msd_t_fmaps
