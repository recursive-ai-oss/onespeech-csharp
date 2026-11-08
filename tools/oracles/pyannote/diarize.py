"""Run the publisher's local community pipeline, writing overlapping RTTM turns."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import argparse
from pathlib import Path
import sys

def main():
    import torch
    import soundfile as sf
    from pyannote.audio import Pipeline
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", type=Path)
    p.add_argument("audio", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=4)
    a = p.parse_args()
    if a.output.exists():
        return
    torch.set_num_threads(a.threads)
    waveform, rate = sf.read(a.audio, dtype="float32", always_2d=True)
    pipeline = Pipeline.from_pretrained(a.snapshot)
    pipeline.to(torch.device("cpu"))
    with torch.inference_mode():
        result = pipeline({"waveform": torch.from_numpy(waveform.T.copy()), "sample_rate": rate})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix(".partial")
    with temp.open("w") as f:
        for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
            f.write(f"SPEAKER {a.audio.stem} 1 {turn.start:.9f} {turn.duration:.9f} <NA> <NA> {speaker} <NA> <NA>\n")
    temp.replace(a.output)
    print(a.audio.stem + " complete", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ": " + str(e), file=sys.stderr)
        sys.exit(1)
