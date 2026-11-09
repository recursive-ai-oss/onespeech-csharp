"""Prepare pinned slice ID manifests and shared PCM arrays, without redistributing corpus audio."""
import warnings
warnings.simplefilter("ignore")  # Do not let dependency warnings print installed-module source lines.
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--earnings", type=Path)
    parser.add_argument("--earnings-count", type=int, default=1)
    parser.add_argument("--only-earnings", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ([] if args.only_earnings else ["test-clean", "test-other"]):
        clips = sorted((args.corpus / split).glob("*/*/*.flac"), key=lambda p: p.stem)
        if args.limit:
            clips = clips[:args.limit]
        target = args.output / split
        target.mkdir(exist_ok=True)
        truths = {}
        for path in clips:
            audio, rate = sf.read(path, dtype="float32")
            assert rate == 16000 and audio.ndim == 1
            audio.astype("<f4").tofile(target / (path.stem + ".pcm"))
            transcript = path.parent / ("-".join(path.stem.split("-")[:2]) + ".trans.txt")
            for line in transcript.read_text().splitlines():
                key, value = line.split(" ", 1)
                truths[key] = value
        (args.output / (split + ".ids")).write_text("".join(path.stem + "\n" for path in clips))
        (args.output / (split + ".truth.json")).write_text(json.dumps({p.stem: truths[p.stem] for p in clips}, indent=2) + "\n")
    if args.earnings:
        if args.earnings_count < 1:
            raise ValueError("earnings-count must be positive")
        paths = sorted(args.earnings.glob("*.mp3"), key=lambda p: (sf.info(p).frames / sf.info(p).samplerate, p.stem))[:args.earnings_count]
        records, truths = [], {}
        for path in paths:
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            # Shared reference/evaluation audio preparation, never a shipped resampler.
            if rate != 16000:
                from scipy.signal import resample_poly
                from math import gcd
                factor = gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // factor, rate // factor).astype("float32")
            pcm = args.output / (path.stem + ".pcm")
            audio.astype("<f4").tofile(pcm)
            with pcm.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"file": path.name, "source_rate": rate, "samples_16000": audio.size, "pcm_sha256": digest})
            text = args.earnings.parent / "transcripts" / "nlp_references" / (path.stem + ".nlp")
            truths[path.stem] = " ".join(line.split("|")[0] for line in text.read_text().splitlines()[1:])
        (args.output / "earnings.ids").write_text("".join(path.stem + "\n" for path in paths))
        (args.output / "earnings.truth.json").write_text(json.dumps(truths) + "\n")
        (args.output / "earnings-audio.json").write_text(json.dumps({"records": records, "license": "CC-BY-SA-4.0", "selection": "shortest complete recordings by duration, ID tie break", "method": "soundfile0.14.0 decode; channel mean; scipy resample_poly when needed"}, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
