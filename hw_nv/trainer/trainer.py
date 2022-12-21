import random
# from pathlib import Path
# from random import shuffle

import PIL
# import pandas as pd
import torch
# import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker, MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            discriminator,
            criterion_gen,
            criterion_disc,
            optimizer_gen,
            optimizer_disc,
            config,
            device,
            dataloaders,
            lr_scheduler_gen=None,
            lr_scheduler_disc=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, discriminator, criterion_gen, criterion_disc, optimizer_gen, optimizer_disc, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler_gen = lr_scheduler_gen
        self.lr_scheduler_disc = lr_scheduler_disc
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "mel_loss", writer=self.writer #todo add losses
        )
        self.melspec = MelSpectrogram(MelSpectrogramConfig)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["wav"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model_gen.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.model_disc.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_gen.train()
        self.model_disc.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm gen", self.get_grad_norm(self.model_gen))
            self.train_metrics.update("grad norm disc", self.get_grad_norm(self.model_disc))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler_gen.get_last_lr()[0]
                )
                # self._log_spectrogram(batch["true_mels"])
                # self._log_spectrogram(batch["gen_mels"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        self.model_gen.eval()
        self.model_disc.eval()
        # self._log_predictions()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train == False:
            raise NotImplementedError

        #training discriminator
        batch["true_mels"] = self.melspec(batch["true_wavs"])
        batch["gen_wavs"] = self.model_gen(batch["true_mels"])
        batch["gen_mels"] = self.melspec(batch["gen_wavs"])

        # if is_train:
        self.optimizer_disc.zero_grad()

        batch["mpd_f_preds"], batch["msd_f_preds"], batch["mpd_t_preds"], batch["msd_t_preds"],
        batch["mpd_f_fmaps"], batch["msd_f_fmaps"], batch["mpd_t_fmaps"], batch["msd_t_fmaps"] \
            = self.model_disc(batch["true_wavs"], batch["gen_wavs"].detach())

        batch["mpd_loss"], batch["msd_loss"] = self.criterion_disc(**batch)
        batch["disc_loss"] = batch["mpd_loss"] + batch["msd_loss"]
        batch["disc_loss"].backward()
        self._clip_grad_norm()
        self.optimizer_disc.step()

        #training generator
        self.optimizer_gen.zero_grad()
        batch["mpd_f_preds"], batch["msd_f_preds"], batch["mpd_t_preds"], batch["msd_t_preds"],
        batch["mpd_f_fmaps"], batch["msd_f_fmaps"], batch["mpd_t_fmaps"], batch["msd_t_fmaps"] \
            = self.model_disc(batch["true_wavs"], batch["gen_wavs"])
        
        batch["adv_loss"], batch["mel_loss"], batch["fm_loss"] = self.criterion_gen(**batch)
        batch["gen_loss"] = batch["adv_loss"] + 2 * batch["fm_loss"] + 45 * batch["mel_loss"]
        batch["gen_loss"].backward()
        self._clip_grad_norm()
        self.optimizer_gen.step()

        for loss in ["mpd_loss", "msd_loss", "disc_loss", "adv_loss", "mel_loss", "fm_loss", "gen_loss"]:
            metrics.update(loss, batch[loss].item())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(self):
        # if self.writer is None:
        #     return
        # for speed in [0.8, 1., 1.3]:
        #     for i, phn in tqdm(enumerate(self.encoded_test)):
        #         mel, path = synthesis(self.model, phn, self.device, self.waveglow, i, speed)
        #         name = f'i={i} s={speed}'
        #         self._log_audio('audio ' + name, path)
        #         image = PIL.Image.open(plot_spectrogram_to_buf(mel.detach().cpu().numpy().squeeze(0).transpose(-1, -2)))
        #         self.writer.add_image('melspec ' + name, ToTensor()(image))
        return



    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.detach().numpy().transpose(-1, -2)))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_audio(self, name, audio_path):
        self.writer.add_audio(name, audio_path, sample_rate=22050)