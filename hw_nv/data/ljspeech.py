import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path
import random

import torchaudio
import torch
from hw_nv.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from .melspec import MelSpectrogram

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, part, max_length, data_dir=None, *args, **kwargs):
        super().__init__()

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index = self._sort_index(self._get_or_load_index(part))

        self._melspec = MelSpectrogram()
        self._max_length = max_length

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self._load_audio(audio_path)

        if audio_wave.shape[1] > self._max_length:
            max_audio_start = audio_wave.shape[1] - self._max_length
            audio_start = random.randint(0, max_audio_start)
            audio_wave = audio_wave[:, audio_start:audio_start + self._max_length]

        melspec = self._melspec(audio_wave)
        return {
            'audio': audio_wave,
            'melspec': melspec,
        }

    def __len__(self):
        return len(self._index)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.95 * len(files))
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
            if i < train_length:
                shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.move(str(fpath), str(self._data_dir / "test" / fpath.name))
        shutil.rmtree(str(self._data_dir / "wavs"))


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists(): # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index

    def _load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        return audio_tensor

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def collate(self, batch):
        res = dict()

        res['melspec'] = torch.nn.utils.rnn.pad_sequence(
            [torch.squeeze(s['melspec']).T for s in batch],
            batch_first=True,
        ).permute(0, 2, 1)

        res['audio'] = torch.nn.utils.rnn.pad_sequence(
            [torch.squeeze(s['audio']).T for s in batch],
            batch_first=True,
        )

        return res
