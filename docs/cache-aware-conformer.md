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
All state is ordinary tensor input/output data, never hidden runtime state.

`NemotronAsrEncoderState` owns fixed-size input/output tensors and warms the
initial and steady shape variants. It uses the caller-owned OneModel execution
API; state resets between utterances. Synthetic tests compare cached execution
with whole-sequence chunk attention, including cache truncation, partial final
chunks and reset. Streaming DSP also equals batch raw features exactly on the
synthetic probes.

These are **building blocks, not a registered Nemotron recognizer**. Publisher
checkpoint padding, prompt projection and end-to-end behavior still need
numerical validation. In particular, the publisher's causal subsampler has
additional end-padding behavior which the current prototype does not yet
reproduce. No model catalog entry, downloader branch, live-session registration
or public recognition path uses these prototypes.

The pinned publisher card lists 80/160/320/560/1120 ms chunks. The released
Transformers 5.22.2 processor rejects the 160 ms setting; its declared supported
lookaheads are 0, 3, 6 and 13. NeMo 3.0.0 loads the prompted checkpoint but its
documented cache-aware simulator rejects that model class. The oracle probes
under `tools/oracles/nemo` record these public-API limitations without inspecting
implementation source.
