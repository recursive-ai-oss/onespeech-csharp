"""Black-box NeMo preprocessor probe. Uses only its documented public module API."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")
import sys
import json
import hashlib
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import nemo
    from omegaconf import OmegaConf
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path, corpus, output = map(Path, sys.argv[1:4])
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path)["preprocessor"], resolve=True)
    cfg.pop("_target_", None)
    if len(sys.argv) > 4:
        cfg["normalize"] = None
    cfg["dither"] = 0.0  # deterministic inference; no noise augmentation
    preprocessor = AudioToMelSpectrogramPreprocessor(**cfg).eval()
    print(json.dumps({"nemo_version": nemo.__version__, "torch": torch.__version__, "preprocessor": cfg}), flush=True)
    inputs = []
    for path in sorted(corpus.rglob("*.flac"))[:5]:
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and pcm.ndim == 1
        inputs.append((path.stem, pcm))
    inputs += [("silence", np.zeros(16000, dtype=np.float32)),
               ("dc", np.full(16000, 0.1, dtype=np.float32)),
               ("short", np.linspace(-0.1, 0.1, 400, dtype=np.float32)),
               ("exact", np.linspace(-0.1, 0.1, 512, dtype=np.float32))]
    with (output / "manifest.tsv").open("w") as manifest:
        for name, pcm in inputs:
            with torch.inference_mode():
                features, length = preprocessor(input_signal=torch.from_numpy(pcm)[None], length=torch.tensor([len(pcm)], dtype=torch.long))
            raw = pcm.astype("<f4").tobytes()
            (output / (name + ".pcm")).write_bytes(raw)
            values = features[0].cpu().numpy().astype("<f4")
            (output / (name + ".mel")).write_bytes(values.tobytes())
            manifest.write(f"{name}\t{len(pcm)}\t{int(length[0])}\t{values.shape[0]}\t{values.shape[1]}\t{hashlib.sha256(raw).hexdigest()}\n")
            print(name, len(pcm), int(length[0]), values.shape, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print installed-package tracebacks or implementation lines.
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
