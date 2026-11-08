"""Encoder activations observed with documented public PyTorch module hooks."""
import warnings
warnings.simplefilter("ignore")
import argparse
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
    torch.backends.mkldnn.enabled = False
    model = whisper.load_model(args.model, device="cpu", download_root=str(args.weights)).float().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    modules = dict(model.named_modules())
    hooks = []
    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        value.detach().cpu().numpy().astype("<f4").tofile(args.output / (name + ".f32"))
    def hook(name):
        return lambda module, inputs, output: save(name, output)
    for name in ["conv1", "conv2"]:
        hooks.append(modules["encoder." + name].register_forward_hook(hook(name)))
    for index in range(model.dims.n_audio_layer):
        hooks.append(modules[f"encoder.blocks.{index}"].register_forward_hook(hook("block" + str(index))))
    hooks.append(modules["encoder.blocks.0"].register_forward_pre_hook(lambda module, inputs: save("position", inputs[0])))
    hooks.append(modules["encoder.ln_post"].register_forward_hook(hook("encoder")))
    mel = np.fromfile(args.features, dtype="<f4").reshape(1, model.dims.n_mels, 3000)
    with torch.no_grad():
        model.encoder(torch.from_numpy(mel))
    for handle in hooks:
        handle.remove()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": reference activation probe failed", file=sys.stderr)
        sys.exit(1)
