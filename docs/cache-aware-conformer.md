# Cache-aware Conformer building blocks

`NemoMelStream` emits raw, unnormalized mel frames with the same arithmetic as
`NemoLogMelSpectrogram.Extract(normalize: false)`. Its fixed ring preserves
preemphasis across pushes and zero-pads centered windows only at stream edges.
It never estimates whole-utterance statistics. Each callback borrows a single
reused 128-value frame; a consumer must finish using it before returning.

The internal cache graph prototypes follow Noroozi et al., *Stateful Conformer
with Cache-based Inference*, §3.3. Every attention layer accepts and returns
projected key/value tensors for 56 past frames. Every depthwise convolution
accepts and returns its previous `kernel_size - 1` GLU activations. The three
stride-two subsampling stages carry their input history and stride phase.
Their frequency padding is two positions on the left and one on the right.
Offline time padding is likewise two past positions and one end position at
each stage: if `f(L) = floor(L/2) + 1`, the encoded length is `f(f(f(L)))`,
not `ceil(L/8)`. The masked final STFT frame is cropped before subsampling.
All state is ordinary tensor input/output data, never hidden runtime state.

`NemotronAsrEncoderState` owns fixed-size input/output tensors and warms the
initial and steady shape variants. A separate end-padding graph consumes the
exact unpadded tail; padding mel input alone would incorrectly evaluate
convolution biases beyond the real tail instead of padding each layer. It uses the caller-owned OneModel execution
API; state resets between utterances. Synthetic tests compare cached execution
with whole-sequence chunk attention, including cache truncation, partial final
chunks and reset. Streaming DSP also equals batch raw features exactly on the
synthetic probes.

These are **building blocks, not a registered Nemotron recognizer**. The
subsampling padding and offline encoder were checked on five fixed clips:
final encoder max absolute error is below `3e-6` versus Transformers 5.22.2
and below `2e-6` versus NeMo 3.0.0 with the same raw features and valid lengths
(check tolerance: `1e-5` maximum absolute error).
This is a float32 numerical parity check, not exact-mode equivalence or a
recognition-quality row. On the same reference encoder outputs, the prompt
projector plus greedy RNN-T decoder matched all 207 nonblank reference token
IDs. A separate 42-frame probe produced exactly equal cached and whole-sequence
encoder output at 80/320/560/1120 ms. These small probes do not establish full
streaming recognition quality or event behavior. No model catalog entry, downloader branch,
live-session registration or public recognition path uses these prototypes.

The pinned publisher card lists 80/160/320/560/1120 ms chunks. The released
Transformers 5.22.2 processor rejects the 160 ms setting; its declared supported
lookaheads are 0, 3, 6 and 13. NeMo 3.0.0 loads the prompted checkpoint but its
documented cache-aware simulator rejects that model class. The oracle probes
under `tools/oracles/nemo` record these public-API limitations without inspecting
implementation source.
