import os

import torch
import torchaudio

from hw_nv.model import HiFiGAN
from hw_nv.data.melspec import MelSpectrogram


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = HiFiGAN().to(device).eval()
    model.load_state_dict(torch.load('eval_model.pth', map_location=device)['state_dict'])

    melspec = MelSpectrogram()

    for fname in os.listdir('test_wavs'):
        wav, sr = torchaudio.load(os.path.join('test_wavs', fname))

        with torch.no_grad():
            mel = melspec(wav)
            res = model(mel.to(device)).cpu()

            torchaudio.save(os.path.join('test_results', fname), res[0], sr)
