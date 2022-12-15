import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel,
        duration_predicted,
        energy_predicted,
        pitch_predicted,
        mel_target,
        duration_predictor_target,
        energy_target,
        pitch_target):

        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(
            torch.log(duration_predicted),
            torch.log(duration_predictor_target.float().clamp(min=1e-8)))

        energy_loss = self.mse_loss(
            energy_predicted,
            energy_target.float())

        pitch_loss = self.mse_loss(
            pitch_predicted,
            pitch_target.float())

        return mel_loss, duration_predictor_loss, energy_loss, pitch_loss
