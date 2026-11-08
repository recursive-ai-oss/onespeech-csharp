"""Black-box public feature APIs only. Never reads an installed module's source."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", choices=["prepare", "whisper", "transformers"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--librispeech", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reference == "prepare":
        import soundfile as sf
        clips = []
        for path in sorted(args.librispeech.glob("*/*/*.flac"))[:4]:
            audio, rate = sf.read(path, dtype="float32")
            assert rate == 16000 and audio.ndim == 1 and audio.size <= 480000
            target = args.output / (path.stem + ".pcm")
            audio.astype("<f4").tofile(target)
            clips.append({"id": path.stem, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pcm_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "license": "CC-BY-4.0", "source": "https://openslr.org/12"})
        # Integer-indexed tone, no RNG; doubles rounded once to float32.
        tone = np.array([0.25 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(16000)], dtype="<f4")
        tone.tofile(args.output / "tone.pcm")
        clips.append({"id": "tone", "generator": "0.25*sin(2*pi*440*i/16000), i=0..15999, rounded to float32", "seed": 0})
        # Crafted padding and dynamic-range edges.
        for name, audio in [("empty", np.zeros(0)), ("one", np.ones(1)), ("exact-window", np.full(400, .1)), ("dc", np.full(16000, .1)), ("clipped", np.array([1., -1.] * 8000))]:
            audio.astype("<f4").tofile(args.output / (name + ".pcm"))
        (args.output / "inputs.json").write_text(json.dumps(clips, indent=2) + "\n")
        return
    import torch
    torch.set_num_threads(4)
    if args.reference == "whisper":
        import whisper
        version = importlib.metadata.version("openai-whisper")
        assert version == "20250625"
    else:
        from transformers import WhisperFeatureExtractor
        version = importlib.metadata.version("transformers")
        assert version == "5.22.2"
    observations = []
    for path in sorted(args.output.glob("*.pcm")):
        audio = np.fromfile(path, dtype="<f4")
        for bins in [80, 128]:
            if args.reference == "whisper":
                padded = np.pad(audio, (0, 480000 - audio.size))
                features = whisper.log_mel_spectrogram(torch.from_numpy(padded), n_mels=bins).numpy()
            else:
                extractor = WhisperFeatureExtractor(feature_size=bins, sampling_rate=16000, hop_length=160, n_fft=400, chunk_length=30)
                features = extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
            assert features.shape == (bins, 3000)
            target = args.output / f"{path.stem}.{bins}.{args.reference}.f32"
            features.astype("<f4").tofile(target)
            observations.append({"clip": path.stem, "bins": bins, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            if path.stem == "tone" and args.reference == "whisper":
                indices = [0, 1, 20, 50, 99, 100, 101, 1500, 2999, 3000, 3001, 30020, bins * 3000 - 1]
                fixture = {"note": "observed in output of openai-whisper 20250625", "bins": bins, "frames": 3000, "indices": indices, "values": [float(features.flat[i]) for i in indices], "generator": "0.25*sin(2*pi*440*i/16000), i=0..15999, rounded to float32", "seed": 0}
                (args.output / f"tone-{bins}.json").write_text(json.dumps(fixture, indent=2) + "\n")
    (args.output / f"{args.reference}-manifest.json").write_text(json.dumps({"version": version, "torch": torch.__version__, "numpy": np.__version__, "observations": observations}, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not print third-party frames or exception source excerpts.
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
