"""Prepare fixed, ordinal LibriSpeech slices; consume only corpus data."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def main():
    import soundfile as sf
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("state", type=Path)
    a = p.parse_args()
    for split, count in [("test-clean", 200), ("test-other", 100)]:
        records = []
        paths = sorted((a.corpus / split).glob("*/*/*.flac"), key=lambda x: x.stem)[:count]
        if len(paths) != count:
            raise ValueError("Incomplete corpus")
        for path in paths:
            key = path.stem
            texts = dict(line.split(" ", 1) for line in path.with_name("-".join(key.split("-")[:2]) + ".trans.txt").read_text().splitlines())
            audio, rate = sf.read(path, dtype="int16")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Expected original mono 16 kHz corpus")
            target = a.state / split / "corpus" / key.split("-")[0]
            target.mkdir(parents=True, exist_ok=True)
            wav = target / (key + ".wav")
            sf.write(wav, audio, rate, subtype="PCM_16")
            wav.with_suffix(".lab").write_text(texts[key] + "\n")
            records.append({"id": key, "text": texts[key], "audio": str(wav.resolve()), "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wavSha256": hashlib.sha256(wav.read_bytes()).hexdigest()})
        (a.state / split / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise SystemExit(type(e).__name__ + ": " + str(e)) from None
