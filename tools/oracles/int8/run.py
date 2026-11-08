"""Black-box released CT2 conversion and faster-whisper int8 evaluation; never inspect modules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["convert", "batch"])
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--count", type=int, default=200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "convert":
        command = [str(Path(sys.executable).with_name("ct2-transformers-converter")), "--model", str(args.model), "--output_dir", str(args.output), "--quantization", "int8", "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise RuntimeError("Released CT2 converter failed; no implementation diagnostics retained")
        manifest = {"converter": "ctranslate2 4.8.5", "quantization": "int8", "publisher_provenance": (args.model / "upstream.txt").read_text().splitlines(), "files": []}
        for file in sorted(args.output.iterdir()):
            if file.is_file():
                with file.open("rb") as stream:
                    manifest["files"].append({"name": file.name, "size": file.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
        (args.output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8", cpu_threads=8, num_workers=1, local_files_only=True)
    for file in sorted(args.split.rglob("*.flac"), key=lambda f: f.stem)[args.begin:args.begin + args.count]:
        target = args.output / (file.stem + ".json")
        if target.exists():
            continue
        audio, rate = sf.read(file, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected original mono 16 kHz LibriSpeech")
        segments, _ = model.transcribe(np.asarray(audio), language="en", task="transcribe", beam_size=1, best_of=1, temperature=0.0, vad_filter=False, word_timestamps=False, condition_on_previous_text=True)
        text = "".join(segment.text for segment in segments)
        temporary = target.with_suffix(".part")
        temporary.write_text(json.dumps({"id": file.stem, "text": text, "oracle": "faster-whisper 1.2.1 / CT2 4.8.5", "precision": "int8", "beam_size": 1, "temperature": 0, "vad_filter": False}) + "\n")
        temporary.replace(target)


if __name__ == "__main__":
    # Suppress all oracle diagnostics, including tracebacks from native public API calls.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            main()
        except Exception as error:
            Path(os.environ["TMPDIR"], "int8-oracle-error.txt").write_text(type(error).__name__ + "\n")
            sys.exit(1)
