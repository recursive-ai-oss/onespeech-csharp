"""Pinned CPU inference through documented Transformers processor/generate APIs only.
Shares this project's locked Transformers 5.22.2 environment. Framework output is discarded.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path


def main():
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForRNNT, AutoProcessor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_dir, corpus, out = map(Path, sys.argv[1:4])
    chunk, begin, count = map(int, sys.argv[4:7])
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    if chunk == -3:
        for path in out.glob("*.json"):
            row = json.loads(path.read_text())
            if "raw_ids" in row:
                row["text"] = processor.decode(row["raw_ids"], skip_special_tokens=True)
                row["decoder"] = "processor-rnnt"
                path.write_text(json.dumps(row), encoding="utf-8")
        return
    if chunk == -2:
        cases = [[], [2, 2, 2], [700, 700], [2, 700, 700, 2], [700, 13087, 700]]
        (out / "decode-probe.json").write_text(json.dumps([{"ids": ids, "processor": processor.decode(ids, skip_special_tokens=True), "batch": processor.batch_decode([ids], skip_special_tokens=True), "ungrouped": processor.tokenizer.decode(ids, skip_special_tokens=True, group_tokens=False)} for ids in cases], ensure_ascii=False))
        return
    if chunk == -1:
        cases = [[i] for i in range(13089)] + [[], [0, 2, 0, 193, 13087], [2, 2, 2], [3, 4, 5, 6], list(range(39))]
        rng = np.random.default_rng(1729)
        cases += [rng.integers(0, 13089, size=20).tolist() for _ in range(1000)]
        (out / "vocabulary.json").write_text(json.dumps([{"ids": ids, "text": processor.decode(ids, skip_special_tokens=True)} for ids in cases], ensure_ascii=False))
        return
    model = AutoModelForRNNT.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).eval()
    if chunk:
        processor.set_num_lookahead_tokens(chunk // 80 - 1)
    for path in sorted(corpus.rglob("*.flac"), key=lambda p: p.stem)[:200][begin:begin + count]:
        target = out / (path.stem + ".json")
        if target.exists():
            continue
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000
        emitted = []
        available = 0.0
        class Streamer:
            def put(self, value):
                emitted.extend((int(x), available) for x in value.flatten().tolist())
            def end(self):
                pass
        if not chunk:
            inputs = processor(pcm, sampling_rate=rate, language="en-US", return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(**inputs)
        else:
            first_samples = processor.num_samples_first_audio_chunk
            first_frames = processor.num_mel_frames_first_audio_chunk
            inputs = processor(pcm[:first_samples], sampling_rate=rate, is_streaming=True, is_first_audio_chunk=True, language="en-US", return_tensors="pt")
            first_features = inputs.input_features[:, :first_frames, :]
            def features():
                nonlocal available
                available = min(first_samples, len(pcm)) / rate
                yield first_features
                mel_frame = first_frames
                start = mel_frame * processor.feature_extractor.hop_length - processor.feature_extractor.n_fft // 2
                while start < len(pcm):
                    end = min(len(pcm), start + processor.num_samples_per_audio_chunk)
                    if end <= start:
                        break
                    # The 80 ms mode's first regular FFT context begins before sample zero.
                    # Supply the same zero extension as the centered feature extractor.
                    segment = np.pad(pcm[max(0, start):end], (max(0, -start), 0))
                    # The released generate API requires full steady chunks, including EOF.
                    segment = np.pad(segment, (0, max(0, processor.num_samples_per_audio_chunk - len(segment))))
                    item = processor(segment, sampling_rate=rate, is_streaming=True, is_first_audio_chunk=False, language="en-US", return_tensors="pt")
                    available = end / rate
                    yield item.input_features
                    mel_frame += processor.num_mel_frames_per_audio_chunk
                    start = mel_frame * processor.feature_extractor.hop_length - processor.feature_extractor.n_fft // 2
            inputs["input_features"] = features()
            with torch.inference_mode():
                generated = model.generate(**inputs, streamer=Streamer())
        ids = (generated.sequences if hasattr(generated, "sequences") else generated).flatten().tolist()
        text = processor.decode(ids, skip_special_tokens=True)
        special = set(processor.tokenizer.all_special_ids) | {13087, 13088}
        tokens = [i for i in ids if i not in special]
        times = [t for i, t in emitted if i not in special]
        temp = target.with_suffix(".partial")
        temp.write_text(json.dumps({"id": path.stem, "text": text, "tokens": tokens, "available": times, "duration": len(pcm) / rate, "raw_ids": ids}), encoding="utf-8")
        temp.replace(target)


if __name__ == "__main__":
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            main()
        except Exception as error:
            out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
            (out / "error.json").write_text(json.dumps({"exception": type(error).__name__, "message": str(error)}))
            sys.exit(1)
