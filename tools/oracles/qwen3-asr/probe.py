"""Released Qwen toolkit public inference API with PyTorch public tensor-output hooks."""
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
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    model = Qwen3ASRModel.from_pretrained(str(args.model), dtype=dtype, device_map="cpu",
                                        attn_implementation="eager", max_inference_batch_size=1, max_new_tokens=256)
    shapes = {}
    counts = {}

    def save(label, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        count = counts.get(label, 0)
        counts[label] = count + 1
        if count != 0:
            return
        if label == "lm-head-output":
            tensor = tensor[:, -1:, :]
        if args.compact and label.startswith("conv2d"):
            shapes[label] = list(tensor.shape)
            return
        values = tensor.detach().cpu()
        if values.is_floating_point():
            array = values.float().numpy().astype("<f4")
        else:
            array = values.long().numpy().astype("<i8")
        array.tofile(args.output / (label + (".f32" if array.dtype.kind == "f" else ".i64")))
        shapes[label] = list(values.shape)

    def encoder_hook(module, inputs, kwargs, output):
        for index, tensor in enumerate(inputs):
            save("encoder-input-" + str(index), tensor)
        for name, tensor in kwargs.items():
            save("encoder-" + name, tensor)
        if isinstance(output, torch.Tensor):
            save("encoder-output", output)
        elif hasattr(output, "last_hidden_state"):
            save("encoder-output", output.last_hidden_state)

    def forward_pre(module, inputs, kwargs):
        for name, tensor in kwargs.items():
            save("forward-" + name, tensor)

    # Standard PyTorch hooks expose tensors only; never inspect a module's implementation.
    audio_hook = model.model.get_submodule("thinker.audio_tower").register_forward_hook(encoder_hook, with_kwargs=True)
    input_hook = model.model.register_forward_pre_hook(forward_pre, with_kwargs=True)
    handles = []
    def intermediate(label):
        def hook(module, inputs, output):
            for index, tensor in enumerate(inputs):
                save(label + "-input-" + str(index), tensor)
            save(label + "-output", output)
        return hook
    for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out", "layers.0.self_attn_layer_norm", "ln_post", "proj1", "proj2"):
        handles.append(model.model.get_submodule("thinker.audio_tower." + name).register_forward_hook(intermediate(name)))
    handles.append(model.model.get_submodule("thinker").register_forward_pre_hook(forward_pre, with_kwargs=True))
    def decoder_pre(module, inputs, kwargs):
        for name, value in kwargs.items():
            if isinstance(value, tuple):
                for i, part in enumerate(value):
                    save("decoder-" + name + "-" + str(i), part)
            else:
                save("decoder-" + name, value)
    handles.append(model.model.get_submodule("thinker.model.layers.0.self_attn").register_forward_pre_hook(decoder_pre, with_kwargs=True))
    handles.append(model.model.get_submodule("thinker.model.norm").register_forward_hook(intermediate("decoder-final-norm")))
    def encoder_attention(module, inputs, kwargs):
        for index, value in enumerate(inputs):
            save("encoder-attention-input-" + str(index), value)
        for key, value in kwargs.items():
            save("encoder-attention-" + key, value)
    handles.append(model.model.get_submodule("thinker.audio_tower.layers.0.self_attn").register_forward_pre_hook(encoder_attention, with_kwargs=True))
    handles.append(model.model.get_submodule("thinker.lm_head").register_forward_hook(intermediate("lm-head")))
    thinker = model.model.get_submodule("thinker")
    generate = thinker.generate
    def observed_generate(*inputs, **kwargs):
        output = generate(*inputs, **kwargs)
        save("generated_ids", output if isinstance(output, torch.Tensor) else output.sequences)
        return output
    thinker.generate = observed_generate
    output_root = args.output
    for audio_path in args.audio:
        args.output = output_root if len(args.audio) == 1 else output_root / audio_path.stem
        args.output.mkdir(parents=True, exist_ok=True)
        shapes.clear()
        counts.clear()
        audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Probe requires 16kHz mono audio")
        audio.astype("<f4").tofile(args.output / "audio.f32")
        results = model.transcribe(audio=(audio, rate), language="English")
        value = dict(precision=args.precision, shapes=shapes, text=results[0].text, language=results[0].language)
        (args.output / "probe.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(dict(id=audio_path.stem, **value)))
    audio_hook.remove()
    input_hook.remove()
    for handle in handles:
        handle.remove()
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Oracle failed: " + type(error).__name__ + ": " + str(error).splitlines()[0])
        raise SystemExit(1) from None
