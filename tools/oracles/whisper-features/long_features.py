"""Continuous STFT window probes from the public log-mel API, without loading a model."""
import warnings
warnings.simplefilter("ignore")
import math
from pathlib import Path
import sys
import numpy as np
import torch
import whisper
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
audio = np.array([(.1 if i < 240000 else .7) * math.sin(2 * math.pi * 437 * i / 16000) for i in range(512040)], dtype="<f4")
audio.tofile(root / "input.pcm")
torch.set_num_threads(4)
for bins in [80, 128]:
    features = whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=bins, padding=480000).numpy()
    for offset in range(0, audio.size, 480000):
        features[:, offset // 160:offset // 160 + 3000].astype("<f4").tofile(root / f"{offset}.{bins}.f32")
