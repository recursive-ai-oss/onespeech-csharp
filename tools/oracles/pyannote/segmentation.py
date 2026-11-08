"""Offline black-box segmentation probes through the model card's public API."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path
import sys

def main():
    import torch
    import numpy as np
    from pyannote.audio import Model
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--corpus", type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=True)
    model = Model.from_pretrained(a.snapshot / "pytorch_model.bin").cpu().float().eval()
    state = model.state_dict()
    (a.output / "tensors.json").write_text(json.dumps({k: list(v.shape) for k, v in state.items()}, indent=2))
    # Analytic signals are reproducible across runtimes; public speech is added separately.
    t = np.arange(160000, dtype=np.float32) / 16000
    probes = [np.zeros(160000, dtype=np.float32), 0.1 * np.sin(2 * np.pi * 440 * t),
              0.1 * np.sin(2 * np.pi * (120 * t + 40 * t * t))]
    if a.corpus:
        import soundfile as sf
        for path in sorted(a.corpus.rglob("*.flac"))[:5]:
            samples, rate = sf.read(path, dtype="float32")
            if rate != 16000 or samples.ndim != 1:
                raise ValueError("Probes require mono 16 kHz speech")
            x = np.zeros(160000, dtype=np.float32)
            x[:min(160000, len(samples))] = samples[:160000]
            probes.append(x)
            print(json.dumps({"probe": len(probes)-1, "source": path.name}), flush=True)
    for i, x in enumerate(probes):
        x.astype("<f4").tofile(a.output / f"input-{i}.f32")
        with torch.inference_mode():
            y = model(torch.from_numpy(x.copy()).reshape(1, 1, -1)).detach().cpu().numpy()
        y.astype("<f4").tofile(a.output / f"output-{i}.f32")
        print(json.dumps({"probe": i, "shape": list(y.shape), "min": float(y.min()), "max": float(y.max())}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Do not expose installed-module frames or implementation lines.
        print(type(e).__name__ + ": " + str(e), file=sys.stderr)
        sys.exit(1)
