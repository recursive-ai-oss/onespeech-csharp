"""Observe the community pipeline's documented progress-hook artifacts and public outputs.

Uses the pipeline call, its `hook` argument (documented under "Monitoring progress" on the
model card) and the documented output attributes. The hook only records what it is handed.
Set ONESPEECH_ORACLE_DEVICE=cuda for the GPU venv (tools/oracles/pyannote-gpu).
Usage: pipeline_probe.py SNAPSHOT AUDIO.wav OUT_DIR [--num-speakers N]
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path


def describe(value):
    import numpy as np
    out = {"type": type(value).__name__}
    if hasattr(value, "data") and hasattr(value, "sliding_window"):  # SlidingWindowFeature
        out["data_shape"] = list(value.data.shape)
        sw = value.sliding_window
        out["window"] = {"start": float(sw.start), "duration": float(sw.duration), "step": float(sw.step)}
    elif isinstance(value, np.ndarray):
        out["shape"] = list(value.shape); out["dtype"] = str(value.dtype)
    elif hasattr(value, "shape"):
        out["shape"] = list(value.shape)
    return out


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    snapshot, audio, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    num = int(sys.argv[sys.argv.index("--num-speakers") + 1]) if "--num-speakers" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    wav, rate = sf.read(audio, dtype="float32", always_2d=True)
    pipeline = Pipeline.from_pretrained(snapshot)
    pipeline.to(torch.device(os.environ.get("ONESPEECH_ORACLE_DEVICE", "cpu")))  # "cuda" only in the approved GPU reference run
    seen = []

    def hook(name, artifact, *args, **kwargs):
        rec = {"step": name, "describe": describe(artifact), "kwargs": sorted(kwargs)}
        seen.append(rec)
        data = artifact.data if hasattr(artifact, "sliding_window") else artifact
        if isinstance(data, (np.ndarray, torch.Tensor)):
            arr = np.asarray(data)
            if arr.dtype.kind in "fiub" and arr.size < 50_000_000 and arr.shape[0] != 32:
                f = out / f"{len(seen):02d}-{name}.npy"
                np.save(f, arr)
                rec["file"] = f.name

    with torch.inference_mode():
        result = pipeline({"waveform": torch.from_numpy(wav.mean(axis=1, keepdims=True).T.copy()), "sample_rate": rate},
                          hook=hook, **({"num_speakers": num} if num else {}))
    rec = {"outputs": [a for a in dir(result) if not a.startswith("_")]}
    with (out / "hyp.rttm").open("w") as f:
        for turn, _, spk in result.speaker_diarization.itertracks(yield_label=True):
            f.write(f"SPEAKER x 1 {turn.start:.6f} {turn.duration:.6f} <NA> <NA> {spk} <NA> <NA>\n")
    with (out / "hyp-exclusive.rttm").open("w") as f:
        for turn, _, spk in result.exclusive_speaker_diarization.itertracks(yield_label=True):
            f.write(f"SPEAKER x 1 {turn.start:.6f} {turn.duration:.6f} <NA> <NA> {spk} <NA> <NA>\n")
    if getattr(result, "speaker_embeddings", None) is not None:
        np.save(out / "speaker_embeddings.npy", np.asarray(result.speaker_embeddings))
    (out / "steps.json").write_text(json.dumps({"steps": seen, "result": rec}, indent=1, default=str))
    print("steps", [s["step"] for s in seen])


if __name__ == "__main__":
    main()
