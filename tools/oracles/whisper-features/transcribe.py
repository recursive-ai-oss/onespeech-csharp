"""Resumable reference hypotheses through public transcribe/detect_language calls."""
import warnings
warnings.simplefilter("ignore")  # Do not let dependency warnings print installed-module source lines.
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["tiny", "small", "turbo"])
    parser.add_argument("inputs", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--temperature-zero", action="store_true", help="Disable stochastic fallback without changing rejection thresholds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = whisper.load_model(args.model, device="cpu", download_root=str(args.weights)).float().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    ids = args.manifest.read_text().splitlines()
    for index in range(args.begin, min(len(ids), args.begin + args.count)):
        key = ids[index]
        output = args.output / (key + ".json")
        if output.exists():
            continue
        audio = np.fromfile(args.inputs / (key + ".pcm"), dtype="<f4")
        torch.manual_seed(args.seed)
        temperatures = (0,) if args.temperature_zero else (0, .2, .4, .6, .8, 1.)
        result = model.transcribe(audio, language="en", task="transcribe", fp16=False, verbose=None, temperature=temperatures, compression_ratio_threshold=2.4, logprob_threshold=-1., no_speech_threshold=.6, condition_on_previous_text=True, word_timestamps=args.words)
        # The public API returns JSON-compatible scalar data; ignore opaque/unused tensors.
        value = {"text": result["text"], "segments": result["segments"], "language": result["language"], "oracle": "openai-whisper20250625", "precision": "fp32", "temperature": list(temperatures), "beam_size": None, "best_of": None, "compression_ratio_threshold": 2.4, "logprob_threshold": -1., "no_speech_threshold": .6, "condition_on_previous_text": True, "word_timestamps": args.words, "seed": args.seed, "threads": args.threads}
        output.with_suffix(".part").write_text(json.dumps(value, ensure_ascii=False) + "\n")
        output.with_suffix(".part").rename(output)
        if index < 50 and args.words:
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=model.dims.n_mels)
            _, probabilities = model.detect_language(mel)
            (args.output / (key + ".language")).write_text(max(probabilities, key=probabilities.get))
        print(key, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
