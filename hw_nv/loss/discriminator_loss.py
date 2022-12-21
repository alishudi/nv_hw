import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, mpd_t_preds, msd_t_preds, mpd_f_preds, msd_f_preds, **batch):

        mpd_loss = 0
        for subdiscr_t_pred, subdiscr_f_pred in zip(mpd_t_preds, mpd_f_preds): #iterating over preds of subdiscriminators
            mpd_loss += torch.mean(torch.square(subdiscr_t_pred - 1) + torch.square(subdiscr_f_pred))

        msd_loss = 0
        for subdiscr_t_pred, subdiscr_f_pred in zip(msd_t_preds, msd_f_preds):
            msd_loss += torch.mean(torch.square(subdiscr_t_pred - 1) + torch.square(subdiscr_f_pred))

        return mpd_loss, msd_loss


