"""Fixed last-window logit diagnostic, using reference public forward APIs."""
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
    parser.add_argument("results", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    reference = json.loads((root / "turbo-earnings-zero/reference/4474955.json").read_text())
    seeks = list(dict.fromkeys(s["seek"] for s in reference["segments"]))
    prefixes = json.loads((root / "long-decision/reference-prefixes.json").read_text())
    if len(seeks) != len(prefixes):
        raise ValueError("Cannot associate observed prefixes with output windows")
    seek = 85238
    prefix = prefixes[seeks.index(seek)]
    window = [s for s in reference["segments"] if s["seek"] == seek]
    before = [t for s in window for t in s["tokens"]][:-1]
    output = root / "final-decision"
    output.mkdir(exist_ok=True)
    tokens = prefix + before
    (output / "tokens.json").write_text(json.dumps({"tokens": tokens, "prefixLength": len(prefix), "seek": seek, "candidates": [51165, 51215]}) + "\n")
    torch.set_num_threads(8)
    model = whisper.load_model("turbo", device="cpu", download_root=str(args.weights)).float().eval()
    audio = np.fromfile(root / "eval-inputs/4474955.pcm", dtype="<f4")
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=128, padding=480000)[:, seek:seek + 3000]
    mel.numpy().astype("<f4").tofile(output / "mel.f32")
    with torch.no_grad():
        encoded = model.encoder(mel.unsqueeze(0))
        logits = model.decoder(torch.tensor([tokens]), encoded)[0, -1]
    logits.numpy().astype("<f4").tofile(output / "reference.logits.f32")
    print("seek", seek, "prefix", len(prefix), "shared generated", len(before), "raw candidates", {t: logits[t].item() for t in [51165, 51215]}, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
