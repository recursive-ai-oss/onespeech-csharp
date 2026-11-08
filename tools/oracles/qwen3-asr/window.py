"""Observe whether the public CPU encoder's prefix depends on frames beyond its advertised window."""
import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel
    torch.set_num_threads(4)
    model = Qwen3ASRModel.from_pretrained(str(args.model), dtype=torch.float32, device_map="cpu", attn_implementation="eager", max_inference_batch_size=1)
    encoder = model.model.get_submodule("thinker.audio_tower")
    data = np.fromfile(args.probe / "encoder-input-0.f32", dtype="<f4").reshape(128, -1)
    features = torch.from_numpy(data.copy())
    changed = features.clone()
    changed[:, 800:] = 0
    with torch.inference_mode():
        original = encoder(features, feature_lens=torch.tensor([features.shape[1]])).last_hidden_state
        modified = encoder(changed, feature_lens=torch.tensor([changed.shape[1]])).last_hidden_state
    print(json.dumps(dict(frames=features.shape[1], prefix_tokens=104,
                          prefix_max_abs=float((original[:104] - modified[:104]).abs().max()),
                          tail_max_abs=float((original[104:] - modified[104:]).abs().max()))))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Oracle failed: " + type(error).__name__ + ": " + str(error).splitlines()[0])
        raise SystemExit(1) from None
