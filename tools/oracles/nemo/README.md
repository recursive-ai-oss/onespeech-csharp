# NeMo reference probes

These are black-box CPU inference drivers for the pinned released environment in
`pyproject.toml` / `uv.lock`. They call only public APIs and never inspect package
implementation files. Nothing here trains, adapts or exports a shipped graph.

- `features.py <model_config.yaml> <corpus> <output> [raw]`: first five sorted
  FLAC files plus synthetic feature edges. Dither is zero.
- `recognize.py <checkpoint.nemo> <corpus> <output> <begin> <count>`: fp32 greedy
  TDT, durations 0–4, maximum 10 symbols per frame, four intra-op CPU threads.
  Inputs are mono 16 kHz arrays. Each result records text, token IDs, emission
  frames and native timestamps; existing completed files are skipped. Use a
  separate output directory for any changed pin or decoding configuration.
  Ranges of at most 200 files are intended for targeted resumable probes.
  With begin 0 and count at most 5, also writes public encoder/preprocessor
  output arrays. Inference mode is reasserted before each forward.
- `tokenizer.py <checkpoint.nemo> <output.json>`: decoded text for every singleton
  vocabulary ID and crafted empty, unknown, special-token and punctuation cases.

The reference checkpoint is the publisher's pinned Parakeet-TDT-v3 `.nemo`
artifact. Verify its revision, SHA-256 and CC-BY-4.0 attribution separately before
running. Never use a third-party reupload or redistribute the reference weights.

The C# tool accepts `parakeet-convert`, `parakeet-probe`, `parakeet-batch`,
`parakeet-vocabulary`, `parakeet-eval` and `parakeet-timing`. Conversion streams
publisher tensor data into local-only runtime artifacts. `parakeet-eval` creates
common-normalizer scorer inputs and checks exact token/frame identity. Score
those inputs with `onespeech-eval score --normalizer en --resamples 1000
--seed 1729`, supplying both hypotheses. Timing summaries use the same word
alignment and normalization as the evaluation tool, with no utterances silently
dropped. Record the commands, model/data hashes, versions and machine details
with each result.

## Nemotron 3 Diarization probes

These scripts execute the released public APIs only, in the environment pinned by
`pyproject.toml` and `uv.lock`. They are evaluation tools, not runtime dependencies.

## Nemotron 3 Diarization

`nemotron.py <publisher.nemo> <mono-16k-audio> <output-directory> [seconds]`
uses the model card's `SortformerEncLabelModel.restore_from` and `diarize` APIs.
It selects the offline-style configuration (264 speaker-cache frames, 40 FIFO
frames, 340 chunk frames, 40 right-context frames, update period 300; each frame
is 80 ms), fp32, CPU, four Torch threads, batch size one. Outputs are speaker
segments in JSON and little-endian float32 activity probabilities. The optional
seconds limit takes a prefix without selecting speech by its predicted output.
Run one recording per invocation to resume an evaluation by output file.

Use only the publisher checkpoint at
`nvidia/Nemotron-3-Diarization`, revision
`f667ed73aee57d40cc39428eb768b4fd87a0a29e`:
`Nemotron-3-Diarization.nemo`, SHA-256
`867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d`.
It is governed by OpenMDW-1.1 and must not be redistributed by this project.

**Compatibility:** the currently pinned NeMo 3.0.0 release rejects this
checkpoint's `self_attention_model='rope'` setting during model loading.
The script exits 1 with a bounded JSON error; no hypotheses or scores are
produced. Do not treat a failed load as a quality measurement or substitute
another architecture. A compatible released oracle is needed before comparison
rows can be generated. Oracle console logging is discarded because some
libraries print implementation tracebacks before propagating an exception.
