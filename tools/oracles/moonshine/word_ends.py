"""Evaluation-only estimated word ends from pinned Whisper tiny's public DTW API."""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path
import sys
import soundfile as sf
import torch
import whisper

try:
    p = argparse.ArgumentParser()
    p.add_argument("split")
    p.add_argument("output")
    p.add_argument("weights")
    p.add_argument("--count", type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(2)
    model = whisper.load_model("tiny", device="cpu", download_root=args.weights).float().eval()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for file in sorted(Path(args.split).rglob("*.flac"), key=lambda f: f.stem)[:args.count]:
        dest = root / (file.stem + ".json")
        if dest.exists():
            continue
        audio, rate = sf.read(file, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected mono 16kHz audio")
        result = model.transcribe(audio, fp16=False, language="en", task="transcribe", temperature=0, verbose=None, word_timestamps=True, condition_on_previous_text=False)
        words = [{"text": w["word"], "start": w["start"], "end": w["end"]} for segment in result["segments"] for w in segment.get("words", [])]
        row = {"id": file.stem, "words": words, "oracle": "openai-whisper20250625", "model": "tiny", "precision": "fp32", "timing": "DTW estimates, not ground truth"}
        dest.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        print(file.stem, flush=True)
except Exception as error:
    print(json.dumps({"error_type": type(error).__name__, "message": str(error).splitlines()[0]}), file=sys.stderr)
    raise SystemExit(2) from None
