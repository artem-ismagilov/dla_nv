import os

import torch
import torchaudio
import yadisk

from hw_nv.model import HiFiGAN
from hw_nv.data.melspec import MelSpectrogram


def load_checkpoint():
    if not os.path.exists('checkpoint_best.pth'):
        print('Downloading checkpoint...')
        yadisk.YaDisk().download_public(
            'https://disk.yandex.ru/d/aWJYcai_WRLh0A',
            'checkpoint_best.pth')

    return torch.load('checkpoint_best.pth', map_location='cpu')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = HiFiGAN(up_channels=512)
    model.load_state_dict(load_checkpoint()['state_dict'])

    model.to(device).eval()

    melspec = MelSpectrogram()

    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    for fname in os.listdir('test_wavs'):
        wav, sr = torchaudio.load(os.path.join('test_wavs', fname))

        with torch.no_grad():
            mel = melspec(wav)
            res = model(mel.to(device)).cpu()

            torchaudio.save(os.path.join('test_results', fname), res[0], sr)
