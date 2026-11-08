"""Prompt diagnostic on exactly the PCM chunks used by the native pipeline.

Uses only the released oracle's documented public API. Results are resumable
per chunk, so a bounded CPU shard never needs to repeat completed inference.
"""
import warnings
warnings.simplefilter("ignore")
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case", choices=["0", "1"])
    parser.add_argument("mode", choices=["baseline", "phrases", "control"])
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = whisper.load_model("small", device="cpu", download_root=str(args.weights)).float().eval()
    folder = args.root / "cases" / args.case
    prompt = None if args.mode == "baseline" else ", ".join((folder / (args.mode + ".txt")).read_text().splitlines())
    outputs = folder / ("reference-whisper-" + args.mode)
    outputs.mkdir(exist_ok=True)
    chunks = sorted((folder / "chunks").glob("*.pcm"))
    if not chunks:
        raise ValueError("Export the pipeline chunks first")
    texts = []
    for chunk in chunks:
        output = outputs / (chunk.stem + ".json")
        if not output.exists():
            audio = np.fromfile(chunk, dtype="<f4")
            result = model.transcribe(audio, language="en", task="transcribe", fp16=False,
                                      verbose=None, temperature=0, initial_prompt=prompt,
                                      condition_on_previous_text=True, word_timestamps=False,
                                      compression_ratio_threshold=2.4, logprob_threshold=-1.,
                                      no_speech_threshold=.6)
            part = output.with_suffix(".partial")
            part.write_text(json.dumps({"text": result["text"]}, ensure_ascii=False))
            part.replace(output)
        texts.append(json.loads(output.read_text())["text"])
        print(chunk.stem, flush=True)
    output = folder / ("reference-whisper-" + args.mode + ".json")
    part = output.with_suffix(".partial")
    part.write_text(json.dumps({"text": " ".join(texts), "oracle": "openai-whisper20250625",
                                "model": "small", "precision": "fp32", "temperature": 0,
                                "initial_prompt": prompt, "condition_on_previous_text": True,
                                "pipeline": "identical native VAD PCM chunks; independent transcribe per chunk",
                                "chunks": len(chunks)}, ensure_ascii=False))
    part.replace(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
