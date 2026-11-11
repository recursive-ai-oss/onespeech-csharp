"""Black-box probe of the community pipeline's published embedding model.

Calls only the documented public API of the released pyannote.audio package:
Model.from_pretrained on a local authorized checkpoint and model(waveforms[, weights]).
Writes float32 inputs and outputs as raw files plus a manifest for the C# comparison.
Usage: embedding_probe.py EMBEDDING_CHECKPOINT AUDIO.wav [AUDIO.wav ...] OUT_DIR
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Model
    checkpoint, *rest = sys.argv[1:]
    out = Path(rest[-1]); out.mkdir(parents=True, exist_ok=True)
    clips = rest[:-1]
    torch.set_num_threads(4)
    model = Model.from_pretrained(checkpoint)
    model.eval()
    manifest = {"documented_attributes": {}, "cases": []}
    for name in ("sample_rate", "dimension", "min_num_samples"):
        value = getattr(model, name, None)
        manifest["documented_attributes"][name] = value if isinstance(value, (int, float, str)) else repr(value)
    try:
        manifest["documented_attributes"]["receptive_field"] = repr(model.receptive_field)
    except Exception as e:
        manifest["documented_attributes"]["receptive_field"] = "error " + type(e).__name__
    try:
        manifest["forward_doc"] = (model.forward.__doc__ or "")[:1500]
    except Exception as e:
        manifest["forward_doc"] = "error"

    def run(label, wav, weights=None):
        rec = {"label": label, "samples": int(wav.shape[-1])}
        x = torch.from_numpy(wav.copy()).reshape(1, 1, -1)
        try:
            with torch.inference_mode():
                if weights is None:
                    y = model(x)
                else:
                    rec["weights_len"] = int(len(weights))
                    y = model(x, weights=torch.from_numpy(np.asarray(weights, dtype=np.float32)).reshape(1, -1))
            y = y.numpy().astype(np.float32).reshape(-1)
            rec["output"] = f"{label}.emb.f32"
            y.tofile(out / rec["output"])
        except Exception as e:  # message is part of the observed behaviour
            rec["error"] = type(e).__name__ + ": " + str(e)[:300]
        (out / f"{label}.in.f32").write_bytes(wav.astype(np.float32).tobytes())
        rec["input"] = f"{label}.in.f32"
        manifest["cases"].append(rec)

    for i, path in enumerate(clips):
        wav, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 16000
        wav = wav.mean(axis=1)
        stem = Path(path).stem
        for sec in (3, 10):
            n = min(len(wav), sec * 16000)
            run(f"{stem}-{sec}s", wav[:n])
        w = wav[:5 * 16000]
        run(f"{stem}-5s-scaled", w * 0.1)
        run(f"{stem}-5s-offset", w + 0.05)
        if i == 0:
            # weights probes: the number of frames the pooling layer expects is unknown, so try several lengths
            for frames in (589, 500, 497, 100):
                run(f"{stem}-10s-ones-w{frames}", wav[:160000], np.ones(frames, dtype=np.float32))
            for frames in (589, 100):
                wt = np.zeros(frames, dtype=np.float32); wt[: frames // 2] = 1
                run(f"{stem}-10s-half-w{frames}", wav[:160000], wt)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done", len(manifest["cases"]))


if __name__ == "__main__":
    main()
