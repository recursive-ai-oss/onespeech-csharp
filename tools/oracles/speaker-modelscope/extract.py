"""Pinned CPU/fp32 black-box speaker embeddings, resumable by utterance file."""
import warnings
import logging
import sys
import json
import argparse
from pathlib import Path
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

MODELS = {
    "camp-en": ("iic/speech_campplus_sv_en_voxceleb_16k", "032b8131a7ad812f87061955ca974c99060c5a03"),
    "camp-zh": ("iic/speech_campplus_sv_zh-cn_16k-common", "013c18884dffb82846cf472dd06dd4a08d97e2cd"),
    "eres2netv2": ("iic/speech_eres2netv2_sv_zh-cn_16k-common", "1cf80d41fb3435bd3d8df185b5c423333b2db42a"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("data")
    parser.add_argument("ids")
    parser.add_argument("output")
    parser.add_argument("start", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--features", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchaudio.compliance.kaldi as kaldi
    torch.set_num_threads(8)
    observed = []
    if args.features:
        original = kaldi.fbank
        def capture(waveform, *positional, **kwargs):
            result = original(waveform, *positional, **kwargs)
            observed.append(result.detach().cpu().clone())
            return result
        kaldi.fbank = capture
    from modelscope.pipelines import pipeline
    source, revision = MODELS[args.model]
    model = pipeline(task="speaker-verification", model=source, model_revision=revision, device="cpu")
    paths = {p.stem: p for split in ["test-clean", "test-other"] for p in (Path(args.data) / split).rglob("*.flac")}
    ids = Path(args.ids).read_text().splitlines()[args.start:args.start+args.count]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    completed = 0
    with torch.inference_mode():
        for utterance in ids:
            target = out / (utterance + ".f32")
            if target.exists():
                continue
            observed.clear()
            value = model.forward(model.preprocess([str(paths[utterance])])).detach().cpu().numpy().reshape(-1)
            if not np.isfinite(value).all() or not np.linalg.norm(value) > 0:
                raise ValueError("Nonfinite/zero embedding")
            temporary = target.with_suffix(".partial")
            value.astype("<f4").tofile(temporary)
            temporary.replace(target)
            if args.features and observed:
                features = observed[0]
                features = features - features.mean(dim=0, keepdim=True)
                features.T.contiguous().numpy().astype("<f4").tofile(out / (utterance + ".features.f32"))
            completed += 1
            print(json.dumps({"id": utterance, "dimension": int(value.size)}), flush=True)
    print(json.dumps({"completed": completed, "model": source, "revision": revision, "precision": "fp32", "device": "cpu"}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)[:1000]}), flush=True)
        sys.exit(1)
