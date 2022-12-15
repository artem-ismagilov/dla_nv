import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from hw_nv.base import BaseTrainer
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker

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
            dataloader,
            lr_schedulers=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizers, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloader
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
            'mel_l1_loss', 'loss_disc_all', 'loss_gen_all', writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            if tensor_for_gpu not in batch:
                continue
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
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

    def _log_audio(self, label, wave, sr):
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

    @staticmethod
    def synthesis(model, WaveGlow, text, fout, alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0):
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().cuda()
        src_pos = torch.from_numpy(src_pos).long().cuda()

        with torch.no_grad():
            mel = model.forward(sequence, src_pos, alpha=alpha, energy_alpha=energy_alpha, pitch_alpha=pitch_alpha)
        mel = mel.contiguous().transpose(1, 2)
        waveglow.inference.inference(mel, WaveGlow, fout)

    @staticmethod
    def get_test_data():
        tests = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest.",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education.",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space."
        ]
        data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)

        return data_list

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
