"""Run the released toolkit's own long-input transcription path on canonical PCM."""
import argparse
import importlib.metadata
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
    parser.add_argument("--threads", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel
    torch.set_num_threads(args.threads)
    model = Qwen3ASRModel.from_pretrained(str(args.model), dtype=torch.float32,
        device_map="cpu", attn_implementation="eager", max_inference_batch_size=1,
        max_new_tokens=4096)
    before = os.getloadavg()
    # Supply the complete recording: chunking and text stitching belong to the oracle.
    audio = np.fromfile(args.audio, dtype="<f4")
    result = model.transcribe(audio=(audio, 16000), language="English", context="")[0]
    args.output.write_text(json.dumps(dict(id="4474955", text=result.text,
        language=result.language, precision="fp32", max_new_tokens=4096, frames=len(audio),
        load_before=before, load_after=os.getloadavg(), threads=args.threads,
        versions={name: importlib.metadata.version(name) for name in
                  ("qwen-asr", "transformers", "torch", "numpy")})) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Oracle failed: " + type(error).__name__ + ": " + str(error).splitlines()[0])
        raise SystemExit(1) from None
