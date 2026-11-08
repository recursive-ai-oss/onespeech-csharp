# Gated models and segmentation

Gated entries are tier 2, explicitly named, never defaults or mirrors. The
catalog pins publisher revisions and each file's SHA-256. `ModelStore.PullAsync`
takes an ephemeral API token, falling back to `ONESPEECH_MODEL_TOKEN` only when
the API argument is null. It checks gate credentials even for cached files.
A missing or whitespace token throws `ModelTokenRequiredException` before I/O.

The HTTP transport sends Bearer credentials only on the initial request to the
catalogued publisher file URL. Mirrors and every redirect target, including
same-host redirects, receive no credentials. Redirects must remain HTTPS, without
userinfo or custom ports, and stop after five hops. HTTP 401/403 responses throw
`ModelTokenRejectedException`, unless the publisher's `X-Error-Code` is
`GatedRepo`, which throws `ModelGateNotAcceptedException`. A server that does not
distinguish a gate refusal from a rejected credential cannot be classified more
precisely. Response bodies, error-message headers, credentials and signed URLs
never enter library exceptions, cache files or `ModelInfo`. No response body is
parsed to guess gate state. Custom `ModelSource` implementations are responsible
for the same confidentiality rules and are trusted host code.

## Local segmentation-3.0

After accepting the publisher's gate and obtaining the pinned snapshot, load it
without network access:

```csharp
using var segmentation = PyannoteSegmentation.LoadLocal(snapshotDirectory);
using var diarizer = new PyannoteDiarizer(segmentation, embedder);
var turns = diarizer.Diarize(audio);
```

These types are in `OneSpeech.Diarization`. The embedder is an independently
loaded 16 kHz `SpeakerEmbedder`, for example CAM++. The diarizer borrows both
models. Neither is a new default. The original checkpoint is read-only and its
SHA-256 must match the catalog's segmentation-3.0 revision. Conversion uses a
bounded inert checkpoint-data reader: no Python objects or code are executed.
Derived graphs and the MIT notice are written only to the specified user cache;
no publisher weights are copied there. `ModelInfo` includes upstream and derived
digests, the publisher attribution and full MIT notice. Derived tier-2 graphs
must not be mirrored, packaged or redistributed.

`GetFrameLogProbabilities` accepts exactly 160000 mono samples at 16000 Hz and
returns 589 × 7 row-major log-probabilities. Class order is silence, speaker 1,
speaker 2, speaker 3, speakers 1+2, 1+3, speakers 2+3, as in the model card. Frame
centers are `(495 + frame * 270) / 16000` seconds. Inference uses float32 OneModel
fast arithmetic, not a claim of bitwise exact mode. Compared with pyannote.audio
4.0.7 / CPU PyTorch 2.11.0, the probability-space max-absolute tolerance is 1e-5
on five fixed LibriSpeech clips and 5e-5 on crafted silence/tone/chirp inputs.
Log-probability errors are reported separately: very small probabilities can
have appreciable log error, especially on constant input after instance norm.
Powerset classes are selected by argmax, with ties going to the first class.

### Graph dataflow

The pinned checkpoint is a PyanNet-style network. Waveform instance
normalization uses the learned affine parameters and population variance with
`epsilon = 1e-5`. Its analytic SincNet bank has 40 real/imaginary pairs of
251-sample filters, stride 10. Low cutoffs are `50 + abs(low_hz_)`; high cutoffs
are `clamp(low + 50 + abs(band_hz_), 50, 8000)` Hz. The checkpoint supplies the
negative half-grid `n_` and half-window `window_`. For bandwidth `b = high-low`,
the real half-filter is `(sin(high*n)-sin(low*n))/(n/2) * window/(2*b)`;
the imaginary half-filter replaces the sine difference with
`cos(low*n)-cos(high*n)`. The real filter is symmetric, center 1; the imaginary
filter is antisymmetric, center 0. Real filters precede imaginary filters.

The first convolution is followed by absolute value. Each of the three
convolutional stages then applies max pooling (kernel/stride 3), affine instance
normalization (`epsilon = 1e-5`), and leaky ReLU (negative slope 0.01). The second
and third convolutions have 60 outputs, kernel 5 and stride 1, with checkpoint
biases. Four bidirectional LSTMs follow, 128 units per direction, concatenating
forward then backward features. Two 128-unit affine layers each use leaky ReLU
(0.01); the seven-unit classifier uses log-softmax. Dropout is inactive at
inference. LSTM gate weights are reordered from checkpoint i/f/g/o to the
OneModel/ONNX i/o/f/c layout without changing learned values.

## Overlapping turns

`PyannoteDiarizer` preserves separate local-speaker masks instead of reducing
them to a speech union. It uses independent 10 s windows, pads the final window,
and clips activity to real audio. Interior frame boundaries are the midpoints
between frame centers, rounded down to a sample; outer boundaries reach the
window edges. Embeddings use concatenated exclusive-speech samples where at
least 100 ms is available, otherwise the speaker's full activity mask, including
overlap. A local speaker with less than 100 ms total activity has no embedding
and produces no attributed turn.

The existing spectral/AHC algorithm clusters these embeddings globally. The
minimum cluster count is at least the largest local-speaker count; contradictory
count hints are refused. Per-window maximum-cosine injective assignment prevents
two local speakers from collapsing onto the same global label. Adjacent turns
of the same speaker are merged; simultaneous turns of different speakers are
preserved. This is our segmentation-plus-clustering policy, **not community-1
equivalence**: community-1 uses another embedder, its supplied PLDA and VBx.
No published PLDA is fitted or reused with incompatible CAM++ embeddings.

`PyannoteSegmenter` also implements `VoiceActivityDetector`. Passing it to
`ClusteringDiarizer` uses only the union of speech and does **not** detect overlap.
Use `PyannoteDiarizer` for overlapping turns.

## Network entry point and community-1

`PyannoteSegmentation.Load` retains its token refusals: without credentials it
throws `ModelTokenRequiredException`; with credentials it currently throws
`FeatureUnavailableException`. It never silently falls back to another model.
Use `LoadLocal` for an authorized segmentation-3.0 snapshot. Local loading does
not attest that a user accepted a gate; obtaining that snapshot is their
responsibility. Community-1 is catalogued and can be explicitly prefetched, but
has no native pipeline adapter. Its reference pipeline is evaluation-only.
