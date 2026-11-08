"""Prepare a pinned 15-clip FLEURS LID probe and run the public reference API.

FLEURS / Conneau et al., Google: CC-BY-4.0; source and attribution retained in
manifest.json. Audio stays in the external evaluation cache, never in the repo.
"""
import warnings
warnings.simplefilter("ignore")
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import whisper

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SOURCES = {
    "en_us": "6428a4d04d3aac29e16b45e039bb1470a8bd7aa334cf92f7984c9c520d1f234d",
    "fr_fr": "99a08ff00edbc7b6f040e23b33e49bc4a1fce38f364d50c603227d612fd797ef",
    "de_de": "3b40c214cabd603f1ddbf3da0499f7780975d6c56b75240f07a8990dbb7a6c07",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=["small", "turbo"], default="small")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    records = []
    for language, digest in SOURCES.items():
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/parquet-data/{language}/test-00000-of-00001.parquet"
        path = args.cache / (language + ".parquet")
        if not path.exists():
            urllib.request.urlretrieve(url, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)
        if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != digest:
            raise ValueError("Dataset SHA-256 mismatch: " + language)
        rows = next(pq.ParquetFile(path).iter_batches(batch_size=5)).to_pylist()
        for index, row in enumerate(rows):
            key = f"{language}-{index}"
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Expected mono 16 kHz")
            pcm = args.cache / (key + ".pcm")
            audio.astype("<f4").tofile(pcm)
            records.append({"id": key, "label": language[:2], "source_id": row["id"], "url": url, "source_sha256": digest, "pcm_sha256": hashlib.file_digest(pcm.open("rb"), "sha256").hexdigest(), "samples": len(audio)})
    manifest = {"dataset": "google/fleurs", "revision": REVISION, "split": "test", "selection": "first five stored rows per language, en_us/fr_fr/de_de", "licence": "CC-BY-4.0", "attribution": "FLEURS, Conneau et al., Google, https://huggingface.co/datasets/google/fleurs", "records": records}
    (args.cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.cache / "ids").write_text("\n".join(r["id"] for r in records) + "\n")
    torch.set_num_threads(8)
    model = whisper.load_model(args.model, device="cpu", download_root=str(args.weights)).float().eval()
    for record in records:
        audio = np.fromfile(args.cache / (record["id"] + ".pcm"), dtype="<f4")
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=model.dims.n_mels)
        _, probabilities = model.detect_language(mel)
        record["prediction"] = max(probabilities, key=probabilities.get)
    (args.cache / "reference.json").write_text(json.dumps({"oracle": "openai-whisper20250625", "model": args.model, "precision": "fp32", "records": records}, indent=2) + "\n")
    print("LID reference", sum(r["prediction"] == r["label"] for r in records), "/", len(records), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
