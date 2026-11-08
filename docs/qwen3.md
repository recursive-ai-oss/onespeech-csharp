# Qwen3 model data and tokenization

The tokenizer loads the published Qwen3-ASR / Qwen3-ForcedAligner `vocab.json`,
`merges.txt` and `tokenizer_config.json`. It does not execute model-repository
code. Supported tokenizer data declares `Qwen2Tokenizer`, with added tokens
that neither strip whitespace nor require word boundaries or normalization.
Each file is limited to 16 MiB; vocabulary IDs and merge counts are below
200,000. Input text is limited to 1 MiB of UTF-8 and invalid UTF-16 is rejected.

Text undergoes Unicode NFC normalization before byte-level BPE. Splitting is
Unicode-scalar-aware, including astral letters. Numeric scalars are separated
individually; contractions are case-insensitive. Whitespace, punctuation and
newline grouping are significant. Merge rank ties are resolved left-to-right.
These behaviors are checked against the public `Qwen2Tokenizer` API in
Transformers 5.22.2, including combining marks, numbers, emoji and newlines.
The implementation does not add BOS, EOS, language or conversation tokens.

Added tokens are recognized atomically, but not all added tokens are special:
`<asr_text>` and the aligner's `<timestamp>` are ordinary added tokens and
remain present when decoding with special tokens omitted. Encoding special
tokens requires explicit permission. Unknown IDs are rejected; decoding
incomplete UTF-8 byte sequences uses the replacement character.

The pinned ASR and aligner preprocessors both request 128-bin Whisper features:
16 kHz mono input, 400-sample periodic Hann windows, 160-sample hops, centered
reflection, Slaney mel filters, power floor 1e-10, log10 peak-minus-eight
clipping, then `(x + 4) / 4`. A variable-length single-input probe uses
`WhisperLogMelSpectrogram(128).Extract(audio, padToWindow: false)`.
It returns mel-major data with `sample_count / 160` frames, dropping the final
STFT frame. Unpadded input must exceed 200 samples; this extractor is bounded
to 30-second windows. The maximum absolute feature tolerance is 3e-5 against
the published feature extractor's float32 output on the synthetic probe set.
This is a feature tolerance, not an encoder or logit equivalence claim.
Long-audio batching and encoder attention windows require separate validation.

## Reproducing observations

`tools/oracles/qwen3/probe.py` runs only the released tokenizer and feature
extractor public APIs, at the versions in its `uv.lock`. It fetches only pinned
JSON/vocabulary/merge data, never weights or remote Python modules. Its output
records tokenizer data SHA-256s, versions, IDs, decoded text and raw float32
feature arrays. `--librispeech` adds the first 200 sorted IDs of test-clean and
test-other; these derived observations stay outside the repository. The
committed fixture contains only synthetic text observations.

Set `ONESPEECH_QWEN_TOKENIZERS` to the probe's tokenizer-data root to run
`Qwen3TokenizerTests` and the optional NativeAOT tokenizer smoke. Set
`ONESPEECH_QWEN_PROBES` to the probe output for the full feature comparison.
`ONESPEECH_QWEN_TOKENIZER_FIXTURE` optionally selects the external transcript
slice observations. No ASR or forced-alignment graph is provided by the
tokenizer API; these checks do not establish transcription or timing accuracy.
