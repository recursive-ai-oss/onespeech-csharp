"""Compare cached live finals with fp32 public generation on the exact retained audio.

Endpoint intervals come from OneSpeech; this is model-output parity, not a
claim of matching an independently released streaming endpoint policy.
"""
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import sys
import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

try:
    model_path, split, native, output = sys.argv[1:]
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    audio_paths = {p.stem: p for p in Path(split).rglob("*.flac")}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for file in sorted(Path(native).glob("*.json")):
        dest = root / file.name
        if dest.exists():
            continue
        data = json.loads(file.read_text())
        audio, rate = sf.read(audio_paths[file.stem], dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected mono 16kHz")
        texts = []
        final_ids = []
        for segment in data["finals"]:
            start, end = round(segment["start"] * rate), round(segment["end"] * rate)
            retained = audio[start:end]
            inputs = processor(retained, sampling_rate=rate, return_tensors="pt")
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=max(1, int(len(retained) / rate * 6.5)))
            token_ids = ids[0].tolist()
            if "ids" in segment and segment["ids"] != token_ids:
                raise ValueError("Final token mismatch for " + file.stem)
            final_ids.append(token_ids)
            texts.append(processor.decode(ids[0], skip_special_tokens=True))
        row = {"id": file.stem, "text": " ".join(texts), "final_ids": final_ids, "oracle": "transformers5.22.2", "precision": "fp32", "comparison": "same final VAD intervals; not a vendor streaming endpoint reference"}
        dest.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        print(file.stem, flush=True)
except Exception as error:
    print(json.dumps({"error_type": type(error).__name__, "message": str(error).splitlines()[0]}), file=sys.stderr)
    raise SystemExit(2) from None
