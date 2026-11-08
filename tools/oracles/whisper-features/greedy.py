"""Greedy token identity through the public decode API on fixed mel arrays."""
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
    parser.add_argument("model")
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = whisper.load_model(args.model, device="cpu", download_root=str(args.weights)).float().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in sorted(args.features.glob(f"1089*.{model.dims.n_mels}.whisper.f32"))[:3]:
            mel = torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(model.dims.n_mels, 3000))
            result = whisper.decode(model, mel, whisper.DecodingOptions(language="en", task="transcribe", temperature=0, fp16=False))
            (args.output / (path.name.split(".")[0] + ".json")).write_text(json.dumps(result.tokens) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": reference greedy call failed", file=sys.stderr)
        sys.exit(1)
