"""Black-box transcription via the model card's public inference pipeline."""
import argparse
import json
import os
from pathlib import Path
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--cache", type=Path, required=True)
    options = parser.parse_args()
    os.environ["XDG_CACHE_HOME"] = str(options.cache.expanduser().resolve())
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M", device=torch.device("cpu"), dtype=torch.float32)
    clips = sorted(options.data.rglob("*.flac"), key=lambda p: p.stem)[: options.count]
    if len(clips) != options.count:
        raise ValueError("Insufficient fixed probe clips")
    existing = set()
    if options.output.exists():
        for line in options.output.read_text().splitlines():
            existing.add(json.loads(line)["id"])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("a", encoding="utf-8") as writer, torch.inference_mode():
        for clip in clips:
            if clip.stem in existing:
                continue
            prediction = pipeline.transcribe([str(clip)], batch_size=1)
            if not isinstance(prediction, list) or len(prediction) != 1 or not isinstance(prediction[0], str):
                raise ValueError("Unexpected public transcription result")
            writer.write(json.dumps({"id": clip.stem, "text": prediction[0], "oracle": "omnilingual-asr/0.2.0", "precision": "fp32"}, ensure_ascii=False) + "\n")
            writer.flush()
            print(clip.stem, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print installed-module frames or source lines.
        print("Oracle failed: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
