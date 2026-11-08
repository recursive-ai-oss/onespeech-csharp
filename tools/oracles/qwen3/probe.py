"""Black-box tokenizer and feature observations; never loads model code or weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

PINS = {
    "asr": ("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
    "aligner": ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
    "asr-1.7b": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--librispeech", type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2Tokenizer, WhisperFeatureExtractor

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    documents = {}
    observations = {}
    texts = ["", " ", "   \t\n\r\n", "Hello world!", "I'm I'M he'd He'LL we've can't",
             "01234567890123456789", "1 23 456 7890 １２３ Ⅻ²", "e\u0301 é naïve",
             "中文 日本語 한국어", "🙂👩🏽\u200d💻🚀", "𐐀𐐨𐐀 test", "\r\n\r\nX\n   \n!\n\n",
             "  foo\tbar\u00a0baz\u2003next", "abc123!!!\n\r last    ", "'S 'T 'RE 'VE 'M 'LL 'D",
             "<|im_start|>assistant\nlanguage English<asr_text>Hello.<|im_end|>",
             "<|audio_start|><|audio_pad|><|audio_end|>", "[time]Hello[time]", "\x00\x01\x7f"]
    # Fixed-seed crafted scalar mixtures exercise regex branch ordering and byte boundaries.
    alphabet = ["a", "Z", "é", "\u0301", "中", "𐐀", "9", "Ⅻ", "²", " ", "\t", "\r", "\n", "!", "'", "🙂", "\u2003", "\u0085"]
    state = 1729
    for _ in range(128):
        text = ""
        for _ in range(96):
            state = (1664525 * state + 1013904223) & 0xffffffff
            text += alphabet[(state >> 16) % len(alphabet)]
        texts.append(text)
    if args.librispeech:
        for split in ("test-clean", "test-other"):
            transcripts = {}
            for path in (args.librispeech / split).rglob("*.trans.txt"):
                for line in path.read_text().splitlines():
                    utterance, text = line.split(" ", 1)
                    transcripts[utterance] = text
            texts.extend(transcripts[key] for key in sorted(transcripts)[:200])
    for key, (repo, rev) in PINS.items():
        directory = args.cache / key
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "preprocessor_config.json"):
            source = Path(hf_hub_download(repo, name, revision=rev))
            data = source.read_bytes()
            (directory / name).write_bytes(data)
            documents[f"{key}/{name}"] = hashlib.sha256(data).hexdigest()
        tokenizer = Qwen2Tokenizer.from_pretrained(directory, local_files_only=True)
        added = json.loads((directory / "tokenizer_config.json").read_text())["added_tokens_decoder"]
        family_texts = texts + [value["content"] for value in added.values()] + ["A" + value["content"] + "B" for value in added.values()]
        observations[key] = [dict(text=text, ids=tokenizer.encode(text, add_special_tokens=False),
                                  decoded=tokenizer.decode(tokenizer.encode(text, add_special_tokens=False),
                                                           skip_special_tokens=False, clean_up_tokenization_spaces=False),
                                  decoded_without_special=tokenizer.decode(tokenizer.encode(text, add_special_tokens=False),
                                                                           skip_special_tokens=True, clean_up_tokenization_spaces=False))
                             for text in family_texts]
    cases = []
    extractor = WhisperFeatureExtractor.from_pretrained(args.cache / "asr", local_files_only=True)
    for name, length in [("silence", 1600), ("impulse", 401), ("dc", 16000),
                         ("noise", 32001), ("exact-window", 480000), ("sub-window", 201)]:
        samples = np.zeros(length, dtype=np.float32)
        if name == "impulse":
            samples[0] = 1
            samples[-1] = -1
        elif name == "dc":
            samples[:] = 0.125
        elif name in ("noise", "exact-window", "sub-window"):
            state = 1729
            for i in range(length):
                state = (1664525 * state + 1013904223) & 0xffffffff
                samples[i] = ((state >> 8) / 8388608.0 - 1.0) * 0.25
        # Qwen's published preprocessor requests an attention mask; inspect padding explicitly.
        result = extractor(samples, sampling_rate=16000, padding="longest", return_attention_mask=True,
                           return_tensors="np")
        features = result.input_features.astype("<f4")
        samples.astype("<f4").tofile(args.output / f"{name}.audio.f32")
        features.tofile(args.output / f"{name}.features.f32")
        cases.append(dict(name=name, samples=length, shape=list(features.shape),
                          mask=result.attention_mask.tolist()))
    result = dict(versions={name: importlib.metadata.version(name) for name in ("transformers", "torch", "numpy")},
                  pins=PINS, documents=documents, tokenizers=observations, features=cases)
    (args.output / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(tokenizer_cases=len(texts), feature_cases=len(cases), versions=result["versions"])))


if __name__ == "__main__":
    # Do not print third-party stack frames or module source on failure.
    try:
        main()
    except Exception as error:
        print(f"Oracle failed: {type(error).__name__}")
        raise SystemExit(1) from None
