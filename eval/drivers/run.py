"""Black-box oracle driver; never imports private APIs or reads installed source."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", choices=["openai-whisper", "transformers", "faster-whisper"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--checkpoint", help="OpenAI checkpoint path to hash; must be the released model selected")
    p.add_argument("--cache", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.start < 0 or args.end <= args.start or args.threads < 1:
        p.error("Invalid range or thread count")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    versions = {n: importlib.metadata.version(n) for n in [args.oracle, "torch", "numpy", "librosa", "soundfile"]}
    expected = {"openai-whisper": "20250625", "transformers": "5.22.2", "faster-whisper": "1.2.1"}
    if versions[args.oracle] != expected[args.oracle]:
        raise ValueError("Oracle version differs from the pinned version")
    if args.oracle == "faster-whisper":
        versions["ctranslate2"] = importlib.metadata.version("ctranslate2")
        if versions["ctranslate2"] != "4.8.5":
            raise ValueError("CTranslate2 version differs from the pin")
    settings = dict(task="transcribe", language="en", temperature=0.0, beam_size=1,
                    condition_on_previous_text=False, word_timestamps=True,
                    no_speech_threshold=None, logprob_threshold=None, compression_ratio_threshold=None)
    header = dict(type="header", schemaVersion=1, oracle=args.oracle, versions=versions,
                  model=args.model, revision=args.revision, device="cpu", precision="float32",
                  threads=args.threads, seed=0, start=args.start, end=args.end,
                  manifestSha256=digest(args.manifest),
                  audio=dict(decoder="soundfile", downmix="arithmetic mean float32", sampleRate=16000,
                             resampler="librosa soxr_hq", normalize=False), decoding=settings)
    if args.oracle == "openai-whisper":
        if not args.checkpoint or not Path(args.checkpoint).is_file():
            p.error("--checkpoint must name the released checkpoint, downloaded before this run")
        header["checkpointSha256"] = digest(args.checkpoint)
        settings.update(fp16=False, best_of=None, beam_size=None, patience=None,
                        length_penalty=None, suppress_tokens="-1", initial_prompt=None,
                        carry_initial_prompt=False, without_timestamps=False, max_initial_timestamp=1.0,
                        prepend_punctuations="\"'“¿([{-", append_punctuations="\"'.。,，!！?？:：”)]}、")
    elif args.oracle == "transformers":
        # Pipeline kwargs and generation kwargs are recorded separately; no generation sampling.
        header["decoding"] = dict(return_timestamps="word", chunk_length_s=30, batch_size=1,
                                  generate_kwargs=dict(task="transcribe", language="en", do_sample=False,
                                                       num_beams=1, max_new_tokens=440))
    else:
        settings["log_prob_threshold"] = settings.pop("logprob_threshold")
        settings.update(best_of=1, patience=1.0, length_penalty=1.0, repetition_penalty=1.0,
                        no_repeat_ngram_size=0, suppress_blank=True, suppress_tokens=[-1],
                        initial_prompt=None, prefix=None, without_timestamps=False,
                        max_initial_timestamp=1.0, vad_filter=False, hotwords=None,
                        prepend_punctuations="\"'“¿([{-", append_punctuations="\"'.。,，!！?？:：”)]}、")
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest has duplicate IDs")
    selected = records[args.start:args.end]
    if not selected or args.end > len(records):
        raise ValueError("Range is outside the manifest")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open() as f:
            if json.loads(next(f)) != header:
                raise ValueError("Resume header differs; use a new output")
            for line in f:
                row = json.loads(line)
                if row["id"] in done or row["id"] not in {r["id"] for r in selected}:
                    raise ValueError("Invalid resume IDs")
                done.add(row["id"])
    if len(done) == len(selected):
        return
    if args.oracle == "openai-whisper":
        import whisper
        model = whisper.load_model(args.checkpoint, device="cpu", download_root=args.cache)
    elif args.oracle == "transformers":
        from transformers import pipeline
        model = pipeline("automatic-speech-recognition", model=args.model, revision=args.revision,
                         device="cpu", dtype=torch.float32, trust_remote_code=False, model_kwargs={"cache_dir": args.cache})
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cpu", compute_type="float32", cpu_threads=args.threads,
                             num_workers=1, download_root=args.cache, revision=args.revision)
    with output.open("a", encoding="utf-8") as out:
        if not done:
            # An empty existing file is not resumable; a header is written exactly once.
            if output.stat().st_size == 0:
                out.write(json.dumps(header, sort_keys=True) + "\n")
                out.flush()
        for record in selected:
            if record["id"] in done:
                continue
            audio, rate = sf.read(record["audio"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1, dtype=np.float32)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if not np.isfinite(audio).all():
                raise ValueError("Non-finite audio")
            if args.oracle == "openai-whisper":
                result = model.transcribe(audio, verbose=None, **settings)
                text = result["text"]
                words = [dict(text=w["word"], start=w["start"], end=w["end"])
                         for s in result["segments"] for w in s.get("words", [])]
            elif args.oracle == "transformers":
                result = model(dict(raw=audio, sampling_rate=16000), **header["decoding"])
                text = result["text"]
                words = [dict(text=w["text"], start=w["timestamp"][0], end=w["timestamp"][1]) for w in result["chunks"]]
                for w in words:
                    if w["start"] is None or w["end"] is None:
                        w["start"] = w["end"] = None
            else:
                segments, _ = model.transcribe(audio, **settings)
                segments = list(segments)
                text = "".join(s.text for s in segments)
                words = [dict(text=w.word, start=w.start, end=w.end) for s in segments for w in (s.words or [])]
            out.write(json.dumps(dict(id=record["id"], text=text, words=words), ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())


if __name__ == "__main__":
    # Do not expose tracebacks into an oracle's installed modules.
    try:
        main()
    except Exception as error:
        print(f"Oracle run failed ({type(error).__name__}); inspect inputs and documented API settings.", flush=True)
        raise SystemExit(2) from None
