# Nemotron diarization reference

This is a black-box inference/evaluation environment, locked to the released
Transformers 5.22.2 wheel, CPU Torch 2.15.0, NumPy 2.5.4, SoundFile 0.14.0 and
librosa 1.0.0. Nothing from this environment is a OneSpeech runtime dependency.
Set `UV_PROJECT_ENVIRONMENT` to an external working directory before
`uv sync --frozen --project tools/oracles/nemotron`.

Only use the pinned publisher assets documented in
[the model guide](../../../docs/nemotron-diarization.md). Fetch/verify them into a
private cache, not this checkout; retain their OpenMDW agreement/origin notices.
The reference needs the publisher's `model.safetensors`, `config.json` and
`processor_config.json` at the same revision. It never enables remote model code.

`reference.py <publisher-directory> <mono-16k-audio> <output-directory> [seconds] [hidden]`
executes the documented processor/model/extract-speaker API in float32, CPU,
batch size one, four Torch threads. It writes little-endian float32 probabilities
and JSON segments/provenance, one recording per invocation. Optional seconds takes
a fixed prefix; optional hidden emits features, hidden states and bare-model head
outputs for isolated parity comparisons. A completed file is a resumption unit.

`cache-probe.py` exercises the documented public cache/postprocessor with numeric
frame identifiers and crafted scores, not synthesized voices or biometric speaker
embeddings. `stream-probe.py` checks the first three chunks of a fixed 90 s prefix
using the offline-style cache/FIFO parameters through the public streaming forward.
All probes use public calls and output tensors only, never implementation inspection.

Warnings are suppressed before imports. Oracle stdout/stderr is discarded because
some dependency loggers print module frames before raising. The outer runners emit
only bounded status/error JSON, no traceback frames. No training, adaptation or GPU
execution is performed. NeMo 3.0.0's incompatible RoPE-load repro is retained in
`../nemo/nemotron.py`; it must not be reported as a successful reference inference.
