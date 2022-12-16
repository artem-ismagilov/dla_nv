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
        self.mp_discriminator = MultiPeriodDiscriminator()
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

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mp_discriminator(true_audio, fake_audio)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.ms_discriminator(true_audio, fake_audio)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

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


def init_conv_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


#------------------------------------------------

import torch.nn.functional as F

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
