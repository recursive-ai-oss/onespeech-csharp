# Cache-aware FastConformer recognition

Nemotron 3.5 ASR loads only by explicit model name,
`nvidia/nemotron-3.5-asr-streaming-0.6b`. It is an upstream-only OpenMDW-1.1
model, never a default or a redistributed weight package. `ModelStore` verifies
the pinned publisher safetensors, configuration and tokenizer files, converts
locally, and verifies every converted output. `ModelInfo` retains the full
agreement, publisher origin and all file digests.

```csharp
var resolved = await ModelStore.Default.PullAsync(
    "nvidia/nemotron-3.5-asr-streaming-0.6b");
using var recognizer = NemotronAsrRecognizer.Load(resolved,
    new NemotronAsrOptions { ChunkMilliseconds = 80, Threads = 1 });
using var vad = new SileroVad();
using var live = recognizer.CreateLiveSession(vad, language: "en");
```

Types are in `OneSpeech.Catalog`, `OneSpeech.FastConformer` and `OneSpeech.Vad`.
`Speech.Listen` and `Speech.TranscribeStream` accept the same explicit model ID.
Their default attention chunk is 320 ms, matching the publisher configuration.
The low-level options also accept 80, 560 and 1120 ms. The released reference
processor rejects 160 ms; the library refuses that option rather than silently
changing the reference's supported settings. Language tags cover the publisher's
transcription-ready and broad-coverage locales and their base-language aliases;
adaptation-only locales are not advertised. Null language selects the automatic
prompt and preserves its emitted language tag. Automatic live prompting resets
for every utterance. Translation and phrase biasing are not supported.

## Features, padding and cache state

`NemoMelStream` emits raw, unnormalized 128-bin mel frames with the same arithmetic
as `NemoLogMelSpectrogram.Extract(normalize: false)`. The fixed ring preserves
preemphasis across pushes. Centered windows use zero extension only at stream
edges; there are no whole-utterance statistics. Callbacks borrow one reused
frame. The valid offline frame count is `floor(samples/160)` at 16 kHz. Crop the
masked final STFT frame before subsampling.

The three stride-two subsampling stages pad frequency two positions on the left
and one on the right. Offline time padding is likewise two past positions and
one end position at each stage. Thus, with `f(L) = floor(L/2) + 1`, the encoded
length is `f(f(f(L)))`, not `ceil(L/8)`. An exact-size offline tail graph applies
padding at every stage; padding mel input alone would wrongly evaluate biases
beyond the real tail.

Cache execution follows Noroozi et al., *Stateful Conformer with Cache-based
Inference*, §3.3. Each attention layer carries projected keys/values for 56 past
frames; each causal depthwise convolution carries its previous eight GLU
activations. Subsampling history preserves the stride phase. State is ordinary
caller-owned tensor input/output data. The initial and steady shapes are warmed
and reused, and state resets between utterances.

The documented streaming reference requires full steady chunks, including EOF.
Live recognition therefore zero-extends the final PCM context to the same chunk
geometry. This never advances the caller's real audio clock. Offline recognition
instead uses exact valid lengths. The emitted words use the native RNN-T 80 ms
frame intervals, with softmax token confidence averaged over a word. A partially
covered final frame is clipped to the real horizon; words wholly beyond it remain
untimed (`TimingMethod.None`), not assigned invented boundary times.

## Live events and bounds

The predictor/joint network accepts at most ten nonblank symbols per encoder
frame and keeps its recurrent state across chunks. RNN-T repeats are preserved;
only declared special/language tokens are stripped from text. A preferred partial
cadence follows the selected attention chunk. The shared controller retains its
200 ms default for other families. The actual delivery clock also depends on the
caller's PCM push size and VAD endpoint policy.

The VAD controller bounds utterances at 30 seconds by default, emits revisioned
immutable partial snapshots and one immutable final before each endpoint, and
uses one integer-sample audio clock. File streaming uses the same live controller.
In-memory offline input above 30 seconds uses VAD-first chunks. Models are
borrowed by live sessions and must outlive them; recognizer/session instances are
not thread-safe. Cancellation is checked at feature, graph and decoding steps.

## Small-probe evidence

On five fixed clips, final encoder max absolute error was below `3e-6` versus
Transformers 5.22.2 and below `2e-6` versus NeMo 3.0.0 using the same raw features
and valid lengths (check tolerance `1e-5`, float32 fast arithmetic, not exact-mode
equivalence). Prompt/predictor/joint decoding matched all 207 nonblank reference
IDs, including special tokens. End-to-end decoding on those clips was
lexical-token-identical at 80, 320 and 560 ms. These probes are not full-corpus
quality claims. The sanitized reference harnesses share the locked Transformers
5.22.2 CPU environment under `tools/oracles/moonshine`.

## Streaming measurement method

The streaming comparison uses fp32 greedy RNN-T with at most ten symbols per
frame, the English prompt, and identical publisher weights. Each input push
contains the selected attention chunk's duration. Native rows exercise the live
controller's actual partial publication on known utterances with speech activity
assumed present; the reference uses its documented streaming generator and
streamer. Both complete at the real input horizon. This isolates recognition
from endpoint detection, which the reference API does not provide.

Word finalisation latency here means stable-word publication: a word closes when
the next word begins, or at EOF for the last word. Match words with the same
English scoring normalizer and report matched/unmatched counts. Measure both
implementations against the same offline native word ends; additionally report
available MFA word ends as a separately named clock. Signed negative differences
are retained because native endpoints have 80 ms resolution. EOF Final delay is
reported separately; this method is not a claim about a vendor endpoint policy
or wall-clock network latency.

`nemotron-asr-metrics <converted> <clip> <output> <chunk-ms> [threads]` measures
three warmed live repetitions; threads defaults to one. RTF includes `Push` and
`Complete` processing, not model load or session startup (reported separately).
Allocation accounting separates immutable emitted payloads and VAD-runtime
allocations from working storage. Non-emitting pushes allocate no working bytes,
but partial snapshot preparation and EOF currently do: an overall zero-working-
allocation claim would be incorrect. Indicative shared-load measurements do not
substitute for isolated release benchmarks.
