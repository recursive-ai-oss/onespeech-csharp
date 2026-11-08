"""Black-box feature, hidden-state, first-logit and tokenizer probes via public APIs."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration, MoonshineStreamingForConditionalGeneration


def run():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("audio")
    p.add_argument("output")
    p.add_argument("--without-attention-mask", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    if Path(args.audio).suffix == ".pcm":
        waveform, rate = np.fromfile(args.audio, dtype="<f4"), 16000
    else:
        waveform, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or waveform.ndim != 1:
        raise ValueError("Probe requires mono 16000 Hz audio")
    waveform.astype("<f4").tofile(dest / "raw.bin")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    cls = MoonshineStreamingForConditionalGeneration if "streaming" in args.model else MoonshineForConditionalGeneration
    model = cls.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    inputs = processor(waveform, sampling_rate=rate, return_tensors="pt")
    if args.without_attention_mask:
        inputs.pop("attention_mask", None)
    # Capture the standalone encoder before a decoder can add positions in-place.
    encoder_output = model.get_encoder()(**inputs, return_dict=True)
    encoder_output.last_hidden_state.detach().cpu().numpy().astype("<f4").tofile(dest / "encoder-standalone.bin")
    output = model(**inputs, decoder_input_ids=torch.tensor([[1]]), output_hidden_states=True, output_attentions=True, return_dict=True)
    shapes = {}
    for name, tensor in dict(inputs).items():
        array = tensor.detach().cpu().numpy()
        shapes[name] = list(array.shape)
        array.astype("<f4" if array.dtype.kind == "f" else "<i8").tofile(dest / (name + ".bin"))
    for group, tensors in (("encoder", output.encoder_hidden_states), ("decoder", output.decoder_hidden_states)):
        for index, tensor in enumerate(tensors):
            array = tensor.detach().cpu().numpy().astype("<f4")
            name = f"{group}-{index}"
            shapes[name] = list(array.shape)
            array.tofile(dest / (name + ".bin"))
    for index, tensor in enumerate(output.encoder_attentions):
        if index == 0:
            a = tensor.detach().cpu().numpy()
            q, k = np.indices(a.shape[-2:])
            outside = (k < q - 16) | (k > q + 4)
            print(json.dumps({"attention_outside_local_max": float(a[0, :, outside].max()) if outside.any() else 0.0, "attention_shape": list(a.shape)}))
        tensor.detach().cpu().numpy().astype("<f4").tofile(dest / f"attention-{index}.bin")
    output.encoder_last_hidden_state.detach().cpu().numpy().astype("<f4").tofile(dest / "encoder-final.bin")
    logits = output.logits.detach().cpu().numpy().astype("<f4")
    shapes["logits"] = list(logits.shape)
    logits.tofile(dest / "logits.bin")
    ids = model.generate(**inputs, do_sample=False, max_new_tokens=max(1, int(len(waveform) / rate * 6.5)))
    result = {"shapes": shapes, "ids": ids[0].tolist(), "text": processor.decode(ids[0], skip_special_tokens=True)}
    (dest / "probe.json").write_text(json.dumps(result, ensure_ascii=False))
    for text in ("", " hello world", "a  b\nC", "café e\u0301", "中文 🙂", "<s></s>"):
        encoded = processor.tokenizer(text)
        print(json.dumps({"text": text, "ids": encoded["input_ids"], "decoded": processor.tokenizer.decode(encoded["input_ids"])}, ensure_ascii=False))
    print(json.dumps({"shapes": shapes, "text": result["text"], "ids": result["ids"]}))

if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "message": str(error).splitlines()[0]}), file=sys.stderr)
        raise SystemExit(2) from None
