"""Information-only public openai-whisper word timestamps on the same clips."""
import argparse
import json
import os
from pathlib import Path
import sys
import warnings
import wave
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")


def main():
    import numpy as np
    import torch
    import whisper
    p = argparse.ArgumentParser()
    p.add_argument("split", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("begin", type=int)
    p.add_argument("count", type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = whisper.load_model("small", device="cpu", download_root=str(a.weights)).float().eval()
    output = a.split / "oracle"
    output.mkdir(exist_ok=True)
    for line in a.split.joinpath("manifest.jsonl").read_text().splitlines()[a.begin:a.begin+a.count]:
        r = json.loads(line)
        path = output / (r["id"] + ".json")
        if path.exists():
            continue
        with wave.open(r["audio"], "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Unexpected evaluation WAV format")
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        torch.manual_seed(0)
        result = model.transcribe(audio, language="en", task="transcribe", fp16=False, verbose=None, temperature=0, word_timestamps=True, condition_on_previous_text=True, compression_ratio_threshold=2.4, logprob_threshold=-1., no_speech_threshold=.6)
        value = {"text": result["text"], "segments": result["segments"], "oracle": "openai-whisper/20250625", "precision": "fp32", "temperature": 0, "seed": 0}
        path.with_suffix(".part").write_text(json.dumps(value) + "\n")
        path.with_suffix(".part").rename(path)
        print(r["id"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ": " + str(e), file=sys.stderr)
        sys.exit(1)
