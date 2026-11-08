# NeMo reference probes

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
