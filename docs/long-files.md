# Long files and cancellation

Use `Speech.Transcribe` when you need a complete document with whole-recording speaker labels. Use `Speech.TranscribeStream` when you want final segments as bounded VAD-first chunks finish. The latter does not perform whole-file speaker clustering. Neither API requires retaining the entire PCM waveform.

Encoded files are decoded and resampled through OneAudio. Whole-recording diarization and recognition require separate bounded passes. Caller-owned encoded streams may be spooled to a temporary seekable file; ensure adequate disk space and dispose or finish enumeration to release it. Do not modify a file while it is being processed. The final document's words, text and turns necessarily grow with output length; constant audio working storage does not mean a constant-size returned document.

File calls have `CancellationToken` overloads, and asynchronous stream calls require explicit cancellation. Cancellation is checked at audio blocks, chunk boundaries and model operations where supported, not by interrupting arbitrary instructions. It throws `OperationCanceledException`. Progress in the low-level pipeline is audio time processed, not a wall-clock ETA. First-use model acquisition can dominate startup; [pre-fetch](models.md) before interactive use.

Streaming enumeration owns its decoder and loaded model resources. Early disposal releases them. Live cancellation discards pending speech instead of silently finalizing it; normal end-of-input flushes. See [live sessions](live-sessions.md).

Standalone `Speech.Align` currently limits input to 60 seconds. Split longer alignment requests yourself and offset every returned time onto the original audio clock; do not concatenate chunk-relative times as if they were recording-relative. Recognition's timing pipeline has its own bounded chunks. Words that could not be aligned retain null boundaries.

Long-form reference equivalence and full working-set benchmark rows remain [pending](results.md). VAD horizon boundaries and chunk-boundary speech can affect output; file streaming is not a promise of token identity with an oracle's whole-file policy.
