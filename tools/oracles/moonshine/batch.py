"""Pinned public API greedy ASR on a deterministic, resumable LibriSpeech slice."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path
import sys
import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration, MoonshineStreamingForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("split")
    parser.add_argument("output")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    cls = MoonshineStreamingForConditionalGeneration if "streaming" in args.model else MoonshineForConditionalGeneration
    model = cls.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    files = sorted(Path(args.split).rglob("*.flac"), key=lambda f: f.stem)[args.begin:args.begin + args.count]
    for file in files:
        destination = root / (file.stem + ".json")
        if destination.exists():
            continue
        audio, rate = sf.read(file, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected mono 16kHz LibriSpeech")
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=max(1, int(len(audio) / rate * 6.5)))
        text = processor.decode(ids[0], skip_special_tokens=True)
        row = {"id": file.stem, "text": text, "ids": ids[0].tolist(), "duration": len(audio) / rate, "oracle": "transformers5.22.2", "precision": "fp32", "tokensPerSecond": 6.5, "attention": "eager", "threads": 2}
        temporary = destination.with_suffix(".part")
        temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.rename(destination)
        print(file.stem, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "message": str(error).splitlines()[0]}), file=sys.stderr)
        raise SystemExit(2) from None
