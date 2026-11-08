# Nemotron 3 Diarization

Speech recognition for .NET. `OneSpeech.Diarization.NemotronDiarizer` is an explicitly selected,
overlap-aware, tier-2 file diarizer for up to eight anonymous speakers. It is never a default.
The [clustering documentation](diarization.md#voice-clustering-and-overlap-limitation) shows the
one-call pipeline replacement. This is diarization, not source separation or identity recognition.

## Checkpoint and loading

Only `nvidia/Nemotron-3-Diarization` revision
`f667ed73aee57d40cc39428eb768b4fd87a0a29e` is accepted. The publisher's float32
`model.safetensors` is 396,954,592 bytes, SHA-256
`c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6`.
`LoadAsync` uses `ModelStore` to fetch and verify this file and build runtime graphs locally,
under the model-cache lock. `Load(ResolvedModel)` loads an already resolved cache only.
Local runtime hashes are fixed registrations, not download or mirror URLs. There is no
executable publisher serialization, remote Python, redistributed tier-2 graph, or default selection.
Publisher origin and the full OpenMDW-1.1 agreement remain in `ModelInfo`.

Instances borrow no audio beyond processing and own disposable CPU sessions. Do not process
concurrently on one instance. `DiarizeBlocks` accepts consecutive 16 kHz blocks and averages
channels; `AudioFileSource` supplies decoded/resampled file blocks. The transcription pipeline
makes a diarization pass and a recognition pass, so it does not keep the waveform in memory.

## Features and graph

The pinned card/config specifies a 31-layer, width-512, eight-head Transformer with pre-block
LayerNorm, exact-form GELU, RoPE and a final LayerNorm. It is not a FastConformer encoder.
Eight successive 128-bin mel frames are stacked and projected to width 512. Features use the
shared deterministic NeMo log-mel extractor without utterance normalization or dither: 16 kHz,
25 ms symmetric Hann window, 10 ms hop, 512-point centered, zero-padded STFT, preemphasis
0.97, Slaney filters and natural-log energies with an additive 2^-24 guard. The centered trailing
frame is retained but masked to zero. Subsampling padding is zero; an encoder group containing
only padding is excluded as an attention key. Inputs shorter than one complete 10 ms hop yield
no turns.

The classifier projects 512 to 192, convolves over time with kernel 3/padding 1 and 1536 output
channels, and rearranges factor-major channels to eight 10 ms frames. ReLU precedes the dense
192-to-192 projection and follows it; the last projection produces eight activity logits. Sigmoid
is applied independently, so simultaneous speakers remain present. Head ordering and activation
placement were checked through the released public bare-model API, not implementation source.

Parity probes use float32 and OneModel fast arithmetic, not its cross-platform bit-identical exact
mode. The targeted short feature/encoder/probability probes use max-absolute tolerances 1e-4.
Long-recording quality is checked by DER/JER rather than promising token- or probability-bit identity
through discrete cache selection.

## Bounded chunking and speaker cache

The offline-style card profile is fixed: speaker cache 264, FIFO 40, new chunk 340,
right context 40, cache update period 300, in 80 ms encoder-frame units. A model window is at
most 684 frames. RoPE positions restart at zero for each model window. The waveform buffer holds
30.4 s plus small STFT edge context; carried left-edge samples make features independent of caller
block boundaries. No VAD removes gaps before diarization: the original audio clock is preserved.

Following Streaming Sortformer §§3.1–3.3, the cache stores projected input frames, not Transformer
KV state. Activity is averaged over each group of eight output frames. Frame scores use
`log(max(0.25, p[s])) + sum(log(max(0.25, 1-p[j])))` for other speakers, with scores disabled below
p[s]=0.5. If a speaker has at least 16 frames whose joint exclusive activity exceeds 0.5,
its nonexclusive frames are excluded before bonuses. Recency adds 0.05 starting at the nominal
264-frame cache boundary, including the first compression when the cache was empty. These
boundary/filter semantics were established with crafted public-cache probes. Per-speaker representation boosts are 24 frames
by 2 log(2), and 48 frames by log(2). The 256 best non-silence slots are grouped by speaker and
original frame order. Learned silence follows represented speakers; unused/invalid slots sort to
the tail. This silence placement was checked with both crafted cache inputs and observed public
cache outputs; placing unused slots in front changes arrival-order identities.

Only bounded audio/features, FIFO/cache and two cached execution plans are retained. Returned
turn/change-point metadata necessarily grows with conversation activity; no duration-independent
bound is claimed for the returned document itself. Embeddings are temporary frame features and
are never persisted by the library.

## Turns, overlap and word assignment

A speaker is active at probability ≥0.5 on the 10 ms grid. Adjacent active frames coalesce per
speaker; intervals may overlap. Endpoints are clipped to actual audio duration. Empty audio
returns no turns. Nonfinite PCM is rejected and other rates require resampling.

`DiarizeActive`/`DiarizeActiveBlocks` additionally retain changes in the highest-probability active
speaker. Relative activity is 1 for that channel and 0.5 for other simultaneous channels, not a
calibrated probability. Exact ties choose the lower channel index. The pipeline uses this view to
apply its ordinary majority-covered-duration rule, `Overlap` flag and `Unassigned` gaps, without
changing words or timing. The returned transcript records the tier-2 model and licence provenance.

## Reference evaluation

The reference is the released Transformers **5.22.2** public Nemotron diarization API, float32,
CPU, batch size one, using the same publisher safetensors and offline-style profile. Oracle
runners are isolated under `tools/oracles/nemotron` with a locked `uv` environment. Their console
output is discarded; only bounded statuses and explicit output artifacts are exposed. NeMo 3.0.0
cannot load this publisher checkpoint's RoPE encoder; its sanitized repro is retained under
`tools/oracles/nemo`. No unreleased source build is substituted.

The four targeted comparisons use AMI ES2004a/ES2004b Mix-Headset and AliMeeting Eval
R8001_M8004_MS801/R8003_M8001_MS801 far-field channel 1, with the existing preparation/reference
labels, no collar, and overlap scored. Both hypotheses are scored by `DiarizationMetrics` in
`onespeech-eval`; a per-file comparison uses a 0.5 percentage-point bound, not an utterance bootstrap.
These labels differ from the publisher card's forced-aligned benchmark protocol, so card benchmark
numbers are not claimed as reproduced.
