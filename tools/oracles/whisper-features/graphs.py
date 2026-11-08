"""Reference graph calls through the released Whisper model's public forward API."""
import warnings
warnings.simplefilter("ignore")  # Do not let dependency warnings print installed-module source lines.
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import whisper


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["tiny", "small", "turbo"])
    parser.add_argument("config", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False
    model = whisper.load_model(args.model, device="cpu", download_root=str(args.weights)).float().eval()
    generation = json.loads((args.config / "generation_config.json").read_text())
    tokens = torch.tensor([[generation["decoder_start_token_id"], generation["lang_to_id"]["<|en|>"], generation["task_to_id"]["transcribe"]]])
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.no_grad():
        for path in sorted(args.inputs.glob("1089*.pcm"))[:3]:
            audio = np.fromfile(path, dtype="<f4")
            mel = whisper.log_mel_spectrogram(torch.from_numpy(np.pad(audio, (0, 480000 - len(audio)))), n_mels=model.dims.n_mels)
            encoded = model.encoder(mel.unsqueeze(0))
            logits = model.decoder(tokens, encoded)
            encoded.cpu().numpy().astype("<f4").tofile(args.output / (path.stem + ".encoder.f32"))
            logits.cpu().numpy().astype("<f4").tofile(args.output / (path.stem + ".logits.f32"))
    weight_manifest = [{"file": path.name, "size": path.stat().st_size, "sha256": digest(path)} for path in args.weights.glob("*.pt")]
    (args.output / "weights.json").write_text(json.dumps(weight_manifest, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
