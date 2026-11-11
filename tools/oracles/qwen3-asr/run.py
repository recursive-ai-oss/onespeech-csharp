"""Resumable fixed LibriSpeech slices using only Qwen's released public inference API."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("test-clean", "test-other"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--mode", choices=("asr", "align"), default="asr")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=1, help="utterances per transcribe call (asr mode)")
    args = parser.parse_args()
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    torch.set_num_threads(4)
    if args.device == "cuda":
        # fp32 reference: no TF32 shortcuts in matmuls or convolutions.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted((args.root / args.split).rglob("*.flac"), key=lambda path: path.stem)
    references = {}
    for path in (args.root / args.split).rglob("*.trans.txt"):
        for line in path.read_text().splitlines():
            key, text = line.split(" ", 1)
            references[key] = text
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    if args.mode == "asr":
        model = Qwen3ASRModel.from_pretrained(str(args.model), dtype=dtype, device_map=args.device, attn_implementation="eager",
                                            max_inference_batch_size=args.batch, max_new_tokens=512)
    else:
        model = Qwen3ForcedAligner.from_pretrained(str(args.model), dtype=dtype, device_map=args.device, attn_implementation="eager")
    metadata = dict(python=platform.python_version(), machine=platform.uname()._asdict(), logical_cpus=os.cpu_count(),
                    threads=torch.get_num_threads(), load_before=os.getloadavg(), precision=args.precision, device=args.device, batch=args.batch,
                    gpu=(torch.cuda.get_device_name(0) if args.device == "cuda" else None),
                    cuda=torch.version.cuda,
                    language="English", context="", max_new_tokens=512, attention="eager", mode=args.mode,
                    versions={name: importlib.metadata.version(name) for name in ("qwen-asr", "transformers", "torch", "numpy", "soundfile")})
    sampled_during = False
    selected = [(index, path) for index, path in enumerate(files[args.begin:args.begin + args.count], start=args.begin)
                if not (args.output / (path.stem + ".json")).exists()]
    step = args.batch if args.mode == "asr" else 1
    for at in range(0, len(selected), step):
        group = selected[at:at + step]
        audios = [sf.read(path, dtype="float32") for _, path in group]
        if args.mode == "asr":
            results = model.transcribe(audio=audios, language="English", context="")
            values = [dict(id=path.stem, text=result.text, language=result.language) for (_, path), result in zip(group, results)]
        else:
            (_, path), = group
            result = model.align(audio=audios[0], text=references[path.stem], language="English")[0]
            values = [dict(id=path.stem, text=references[path.stem], words=[dict(text=item.text, start=item.start_time, end=item.end_time) for item in result])]
        if not sampled_during:
            metadata["load_during"] = os.getloadavg()
            sampled_during = True
        for (index, path), value in zip(group, values):
            target = args.output / (path.stem + ".json")
            target.with_suffix(".partial").write_text(json.dumps(value, ensure_ascii=False) + "\n")
            target.with_suffix(".partial").rename(target)
            print(f"{index}: {path.stem}", flush=True)
    metadata["load_after"] = os.getloadavg()
    (args.output / f"metadata-{args.begin}.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Oracle failed: " + type(error).__name__ + ": " + str(error).splitlines()[0])
        raise SystemExit(1) from None
