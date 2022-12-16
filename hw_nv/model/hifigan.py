import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn.utils import weight_norm, spectral_norm, remove_weight_norm
from typing import List

from hw_nv.data.melspec import MelSpectrogram


class HiFiGANConfig:
    upsample_rates: List[int] = [8, 8, 2, 2]
    upsample_kernels: List[int] = [16, 16, 4, 4]
    up_channels: int = 128
    residual_kernel_sizes: List[int] = [3, 7, 11]
    residual_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


class HiFiGAN(nn.Module):
    def __init__(self, config = HiFiGANConfig()):
        super().__init__()

        self.generator = _Generator(config)
        self.mp_discriminator = _MPDiscriminator()
        self.ms_discriminator = MultiScaleDiscriminator()

        self.melspec = MelSpectrogram()

    def forward(self, x):
        return self.generator(x)

    def optimization_step(self, batch, optimizer_g, optimizer_d):
        true_melspec = batch['melspec'].to(self.device)

        fake_audio = self.generator(true_melspec)
        true_audio = batch['audio'].to(self.device).unsqueeze(1)

        losses = self._dicsriminator_step(optimizer_d, true_melspec, true_audio, fake_audio)
        losses.update(self._generator_step(optimizer_g, true_melspec, true_audio, fake_audio))

        return losses

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _dicsriminator_step(self, optimizer_d, true_melspec, true_audio, fake_audio):
        optimizer_d.zero_grad()

        def get_loss(true_out, fake_out):
            loss = 0
            for t, f in zip(true_out, fake_out):
                loss += torch.mean((t - 1) ** 2) + torch.mean(f ** 2)
            return loss

        true_out, fake_out, _, _ = self.mp_discriminator(true_audio, fake_audio.detach())
        loss_mp = get_loss(true_out, fake_out)

        true_out, fake_out, _, _ = self.ms_discriminator(true_audio, fake_audio.detach())
        loss_ms = get_loss(true_out, fake_out)

        loss_all = loss_ms + loss_mp

        loss_all.backward()
        optimizer_d.step()

        return {
            'discriminator_loss_all': loss_all.item(),
            'discriminator_loss_mp': loss_mp.item(),
            'discriminator_loss_ms': loss_ms.item(),
        }

    def _generator_step(self, optimizer_g, true_melspec, true_audio, fake_audio):
        optimizer_g.zero_grad()

        fake_melspec = self.melspec(fake_audio.squeeze())

        loss_mel = F.l1_loss(true_melspec, fake_melspec) * 45

        mp_true_out, mp_fake_out, mp_true_features, mp_fake_features = self.mp_discriminator(true_audio, fake_audio)
        ms_true_out, ms_fake_out, ms_true_features, ms_fake_features = self.ms_discriminator(true_audio, fake_audio)

        def features_loss(true_features, fake_features):
            loss = 0
            for ts, fs in zip(true_features, fake_features):
                for t, f in zip(ts, fs):
                    loss += F.l1_loss(t, f)
            return 2 * loss

        mp_features_loss = features_loss(mp_true_features, mp_fake_features)
        ms_features_loss = features_loss(ms_true_features, ms_fake_features)

        def generator_loss(fake_out):
            loss = 0
            for p in fake_out:
                loss += torch.mean((1 - p) ** 2)
            return loss

        loss_gen_f = generator_loss(mp_fake_out)
        loss_gen_s = generator_loss(ms_fake_out)

        loss_gen_all = loss_gen_s + loss_gen_f + mp_features_loss + ms_features_loss + loss_mel

        loss_gen_all.backward()
        optimizer_g.step()

        return {
            'mel_l1_loss': loss_mel.item(),
            'generator_loss_all': loss_gen_all.item(),
        }


class _GeneratorBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()

        convs1 = []
        for i, d in enumerate(dilations):
            p = ((kernel_size - 1) * d) // 2
            convs1.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p)),
                )
            )
        self.convs1 = nn.ModuleList(convs1)

        convs2 = []
        for i in range(len(self.convs1)):
            convs2.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, padding='same')),
                )
            )
        self.convs2 = nn.ModuleList(convs2)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_new = conv1(x)
            x_new = conv2(x_new)
            x = x + x_new

        return x


class _Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.first_conv = weight_norm(nn.Conv1d(80, config.up_channels, 7, 1, padding=3))

        self.up_convs = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        c = config.up_channels
        for r, k in zip(config.upsample_rates, config.upsample_kernels):
            self.up_convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.ConvTranspose1d(c, c // 2, k, r, padding=((k - r) // 2))),
                )
            )

            residuals = nn.ModuleList()
            for k, d in zip(config.residual_kernel_sizes, config.residual_dilations):
                residuals.append(_GeneratorBlock(c // 2, k, d))
            self.residual_blocks.append(residuals)

            c = c // 2

        self.last_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(c, 1, 7, 1, padding=3)),
        )

        self.apply(init_conv_weights)

    def forward(self, x):
        x = self.first_conv(x)

        for up, res in zip(self.up_convs, self.residual_blocks):
            x = up(x)

            x_new = torch.zeros_like(x)
            for r in res:
                x_new += r(x)
            x = x_new / len(res)

        x = self.last_conv(x)
        return torch.tanh(x)


class _PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()

        self.layers = nn.ModuleList()

        prev_c = 1
        c = 32
        for i in range(4):
            self.layers.append(nn.Sequential(
                weight_norm(nn.Conv2d(prev_c, c, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.1),
            ))
            prev_c = c
            c *= 2
        self.layers.append(nn.Sequential(
            weight_norm(nn.Conv2d(prev_c, prev_c, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            nn.LeakyReLU(0.1),
        ))
        self.layers.append(weight_norm(nn.Conv2d(prev_c, 1, kernel_size=(3, 1), padding=(1, 0))))

        self.period = period

    def forward(self, x):
        if x.shape[2] % self.period > 0:
            p = self.period - (x.shape[2] % self.period)
            x = torch.nn.functional.pad(x, (0, p), 'reflect')

        x = x.view(x.shape[0], 1, -1, self.period)
        features = []
        for l in self.layers:
            x = l(x)
            features.append(x)

        return x.view(x.shape[0], 1, -1), features


class _MPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        periods = [2, 3, 5, 7, 11]
        self.submodules = nn.ModuleList([_PeriodDiscriminator(p) for p in periods])

    def forward(self, true_audio, fake_audio):
        true_out, fake_out, true_features, fake_features = [], [], [], []

        for s in self.submodules:
            to, tf = s(true_audio)
            true_out.append(to)
            true_features.append(tf)

            fo, ff = s(fake_audio)
            fake_out.append(to)
            fake_features.append(ff)

        return true_out, fake_out, true_features, fake_features


def init_conv_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


#------------------------------------------------

import torch.nn.functional as F

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
