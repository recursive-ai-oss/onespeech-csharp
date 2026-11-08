"""Capture initial decoder prefixes with documented PyTorch public hooks."""
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
    parser.add_argument("pcm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    model = whisper.load_model("turbo", device="cpu", download_root=str(args.weights)).float().eval()
    prefixes = []
    def capture(module, inputs):
        tokens = inputs[0]
        if tokens.shape[-1] > 1:
            prefixes.append(tokens[0].tolist())
    handle = model.decoder.register_forward_pre_hook(capture)
    audio = np.fromfile(args.pcm, dtype="<f4")
    with torch.no_grad():
        model.transcribe(audio, language="en", task="transcribe", fp16=False, verbose=None, temperature=(0,), compression_ratio_threshold=2.4, logprob_threshold=-1., no_speech_threshold=.6, condition_on_previous_text=True, word_timestamps=False)
    handle.remove()
    args.output.write_text(json.dumps(prefixes) + "\n")
    print("captured prefixes", len(prefixes), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
