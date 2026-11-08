"""Public forced-aligner call; export tensor observations through public PyTorch hooks."""
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
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text().strip()
    if args.text is None:
        raise ValueError("Provide --text or --text-file")
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    model = Qwen3ForcedAligner.from_pretrained(str(args.model), dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    shapes = {}
    def save(name, tensor):
        if name in shapes or not isinstance(tensor, torch.Tensor):
            return
        values = tensor.detach().cpu()
        array = values.float().numpy().astype("<f4") if values.is_floating_point() else values.long().numpy().astype("<i8")
        array.tofile(args.output / (name + (".f32" if array.dtype.kind == "f" else ".i64")))
        shapes[name] = list(values.shape)
    def pre(module, inputs, kwargs):
        for key, value in kwargs.items():
            save(key, value)
    def head(module, inputs, output):
        save("classifier-input", inputs[0])
        save("classifier-output", output)
    model.model.register_forward_pre_hook(pre, with_kwargs=True)
    # The final head name is taken from the released tensor headers.
    model.model.get_submodule("thinker.lm_head").register_forward_hook(head)
    def embedding_hook(module, inputs, output):
        save("input_ids", inputs[0])
    def attention_pre(module, inputs, kwargs):
        for name, tensor in kwargs.items():
            save("attention-" + name, tensor)
    model.model.get_submodule("thinker.model.embed_tokens").register_forward_hook(embedding_hook)
    model.model.get_submodule("thinker.model.layers.0.self_attn").register_forward_pre_hook(attention_pre, with_kwargs=True)
    audio, rate = sf.read(args.audio, dtype="float32")
    audio.astype("<f4").tofile(args.output / "audio.f32")
    results = model.align(audio=(audio, rate), text=args.text, language="English")
    items = [dict(text=item.text, start=item.start_time, end=item.end_time) for item in results[0]]
    (args.output / "probe.json").write_text(json.dumps(dict(items=items, shapes=shapes), indent=2) + "\n")
    print(json.dumps(dict(items=items, shapes=shapes)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Oracle failed: " + type(error).__name__ + ": " + str(error).splitlines()[0])
        raise SystemExit(1) from None
