import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker
from hw_nv.data.melspec import MelSpectrogram

import os
import numpy as np


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizers,
            config,
            device,
            train_dataloader,
            val_dataloader,
            lr_schedulers=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizers, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_schedulers = lr_schedulers
        self.log_step = 15

        self.train_metrics = MetricTracker(
            'mel_l1_loss',
            'generator_loss_all',
            'discriminator_loss_all',
            'discriminator_loss_mp',
            'discriminator_loss_ms',
            writer=self.writer
        )

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

        batch_idx = 0
        for batch in tqdm(self.train_dataloader):
            losses = self.model.optimization_step(
                batch,
                self.optimizer['optimizer_g'],
                self.optimizer['optimizer_d'],
            )

            for name, l in losses.items():
                self.train_metrics.update(name, l)
            loss = losses['mel_l1_loss']

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss
                    )
                )
                self.writer.add_scalar(
                    "learning rate", next(iter(self.lr_schedulers.values())).get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break

            batch_idx += 1

        for s in self.lr_schedulers.values():
            s.step()

        res = self._get_inference_results()
        for label, wav in res.items():
            self._log_audio(label, wav)

        log = last_train_metrics

        return log

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, label, wave, sr=22050):
        self.writer.add_audio(label, torch.tensor(wave), sr)

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

    @torch.no_grad()
    def _get_inference_results(self):
        melspec = MelSpectrogram()

        self.model.eval()

        res = dict()
        for fname in os.listdir('test_wavs'):
            wav, sr = torchaudio.load(os.path.join('test_wavs', fname))

            assert sr == 22050

            label = os.path.splitext(fname)[0]
            res[label + '_original'] = wav

            with torch.no_grad():
                mel = melspec(wav)
                processed_wav = self.model(mel.to(self.device)).cpu().squeeze()

                res[label + '_processed'] = processed_wav.clone()

        return res

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
