import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    """
    Class for calculation of generator losses
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, mpd_f_preds, msd_f_preds, gen_mels, true_mels, mpd_f_fmaps, msd_f_fmaps,
                mpd_t_fmaps, msd_t_fmaps, **batch):
        """
        Calculates generator losses for a given batch
        :return: GAN loss, Mel-Spectrogram loss, Feature Matching loss
        """
        adv_loss = 0
        for subdiscr_preds in mpd_f_preds + msd_f_preds: 
            #discriminators return list of preds of all subdiscriminators instead of sum
            #so need to iterate over them and sum the losses
            adv_loss += torch.mean(torch.square(subdiscr_preds - 1))
        
        mel_loss = self.l1_loss(gen_mels, true_mels)

        gen_fmaps, true_fmaps = mpd_f_fmaps + msd_f_fmaps, mpd_t_fmaps + msd_t_fmaps
        fm_loss = 0
        for subdiscr_f_fmap, subdiscr_t_fmap in zip(gen_fmaps, true_fmaps):
            for f_layer, t_layer in zip(subdiscr_f_fmap, subdiscr_t_fmap):
                fm_loss += self.l1_loss(f_layer, t_layer)

        return adv_loss, mel_loss, fm_loss


