"""Pinned NeMo public-API inference probe; no package implementation inspection."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import nemo
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_path, corpus, output = map(Path, sys.argv[1:4])
    begin, count = map(int, sys.argv[4:6])
    output.mkdir(parents=True, exist_ok=True)
    model = ASRModel.restore_from(str(model_path), map_location=torch.device("cpu")).float().eval()
    cfg = OmegaConf.create({"strategy": "greedy", "greedy": {"max_symbols": 10}, "model_type": "tdt", "durations": [0, 1, 2, 3, 4]})
    model.change_decoding_strategy(cfg)
    print(json.dumps({"nemo": nemo.__version__, "torch": torch.__version__, "precision": "fp32", "decoding": OmegaConf.to_container(cfg)}), flush=True)
    files = sorted(corpus.rglob("*.flac"))[begin:begin + count]
    for path in files:
        result_path = output / (path.stem + ".json")
        if result_path.exists():
            continue
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and pcm.ndim == 1
        model.eval()
        with torch.inference_mode():
            if begin == 0 and count <= 5:
                features, lengths = model.preprocessor(input_signal=torch.from_numpy(pcm)[None], length=torch.tensor([len(pcm)]))
                encoded, encoded_lengths = model.encoder(audio_signal=features, length=lengths)
                features.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".mel"))
                encoded.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".encoded"))
                print(path.stem, list(features.shape), list(encoded.shape), int(encoded_lengths[0]), flush=True)
            hypothesis = model.transcribe([pcm], batch_size=1, return_hypotheses=True, timestamps=True)[0]
        temporary = result_path.with_suffix(".partial")
        temporary.write_text(json.dumps({"id": path.stem, "text": hypothesis.text, "tokens": hypothesis.y_sequence.tolist(), "timestamp": hypothesis.timestamp}, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
        temporary.replace(result_path)
        print(path.stem, hypothesis.text, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
