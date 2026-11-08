"""Black-box public-API probes. Never inspect installed package files or code."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = load_silero_vad(onnx=True)
    paths = sorted(args.data.glob("**/*.flac"))[:3]
    if len(paths) != 3:
        raise RuntimeError("Need three LibriSpeech test-clean FLAC inputs")
    probes = []
    for path in paths:
        data, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and data.ndim == 1
        probes.append((path.stem, data, rate))
    # Exact decimation here is an evaluation input derivation, not a resampler parity claim.
    probes.append((paths[0].stem + "-8k", probes[0][1][::2].copy(), 8000))
    probes.append(("synthetic-silence-noise", np.concatenate((np.zeros(16000, np.float32),
        np.random.default_rng(719).uniform(-0.1, 0.1, 16000).astype(np.float32))), 16000))
    rows = []
    for name, data, rate in probes:
        window = 512 if rate == 16000 else 256
        model.reset_states()
        probabilities = []
        for start in range(0, len(data), window):
            frame = np.pad(data[start:start + window], (0, max(0, start + window - len(data))))
            probabilities.append(float(model(torch.from_numpy(frame), rate).item()))
        intervals = get_speech_timestamps(torch.from_numpy(data), model, sampling_rate=rate,
            threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=250,
            min_silence_duration_ms=100, speech_pad_ms=30, return_seconds=False)
        filename = name + ".f32"
        data.astype("<f4").tofile(args.out / filename)
        rows.append(dict(id=name, file=filename, rate=rate, frames=len(data), probabilities=probabilities, intervals=intervals))
    metadata = {name: importlib.metadata.version(name) for name in ("silero-vad", "torch", "onnxruntime", "numpy", "soundfile")}
    (args.out / "reference.json").write_text(json.dumps(dict(versions=metadata, probes=rows), indent=2))
    print(json.dumps(dict(versions=metadata, clips=[r["id"] for r in rows])))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
