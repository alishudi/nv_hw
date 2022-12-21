import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker


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
        super().__init__(model, criterion, optimizer, config, device)
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
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "mel_loss", writer=self.writer #todo add losses
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        #todo fix
        batch["src_seq"] = torch.stack([sample.long() for sample in batch["src_seq"]]).to(device).squeeze(0)
        batch["mel_target"] = torch.stack([sample.float() for sample in batch["mel_target"]]).to(device).squeeze(0)
        batch["duration"] = torch.stack([sample.int() for sample in batch["duration"]]).to(device).squeeze(0)
        batch["mel_pos"] = torch.stack([sample.long() for sample in batch["mel_pos"]]).to(device).squeeze(0)
        batch["src_pos"] = torch.stack([sample.long() for sample in batch["src_pos"]]).to(device).squeeze(0)
        batch["mel_max_len"] = batch["mel_max_len"][0]
        batch["energy"] = torch.stack([sample.float() for sample in batch["energy"]]).to(device).squeeze(0)
        batch["pitch"] = torch.stack([sample.float() for sample in batch["pitch"]]).to(device).squeeze(0)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
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
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                # self._log_spectrogram(batch["mel"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        self.model.eval()
        # self._log_predictions()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        mel_output, duration_predictor_output, energy_predictor_output, pitch_predictor_output = self.model(**batch)
        batch["mel"] = mel_output
        batch["duration_predicted"] = duration_predictor_output
        batch["energy_predicted"] = energy_predictor_output
        batch["pitch_predicted"] = pitch_predictor_output
        batch["mel_loss"], batch["duration_loss"], batch["energy_loss"], batch["pitch_loss"] = self.criterion(**batch)
        batch["energy_loss"] *= 0.01
        batch["loss"] = batch["mel_loss"] + batch["duration_loss"] + batch["energy_loss"] + batch["pitch_loss"]
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                if self.config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                    self.lr_scheduler.step(batch["loss"].item())
                else:
                    self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("duration_loss", batch["duration_loss"].item())
        metrics.update("energy_loss", batch["energy_loss"].item())
        metrics.update("pitch_loss", batch["pitch_loss"].item())
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
        if self.writer is None:
            return
        for speed in [0.8, 1., 1.3]:
            for i, phn in tqdm(enumerate(self.encoded_test)):
                mel, path = synthesis(self.model, phn, self.device, self.waveglow, i, speed)
                name = f'i={i} s={speed}'
                self._log_audio('audio ' + name, path)
                image = PIL.Image.open(plot_spectrogram_to_buf(mel.detach().cpu().numpy().squeeze(0).transpose(-1, -2)))
                self.writer.add_image('melspec ' + name, ToTensor()(image))



    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.detach().numpy().transpose(-1, -2)))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
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