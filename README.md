# Vocoder homework

## Setup

Tested on Ubuntu + Python 3.8.10
1. Create venv: `python3 -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install dependencies: `pip3 install -r requirements.txt`

## Reproduce

Train for 79 epochs: `python3 train.py -c hw_nv/configs/config2.json`

## Test

Run `python3 test.py`. This will download the checkpoint and run inference for wavs in `test_wavs` directory. The results will be saved to `test_results` directory.
