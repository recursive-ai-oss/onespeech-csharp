# Speaker identity and scores

`SpeakerEmbedder` is the synchronous CPU extraction role. Its model metadata identifies the
checkpoint and licence; `Dimension` fixes output length and `SampleRate` fixes mono input Hz.
`Embed` writes into caller-owned storage and accepts cancellation. Input must be nonempty,
mono, and already resampled. Implementations return finite, nonzero vectors and reject unsupported
input rather than silently changing channels. Callers dispose model resources after consumers stop.
`CampPlusEmbedder` loads a locally converted CAM++ graph; see below.

## Arithmetic

`EmbeddingMath.Cosine` accumulates in double precision and clamps rounding error to [-1,1].
Both vectors must be finite, nonempty, equal-dimensional and nonzero. `Normalize` supports
in-place normalization. `Average` first unit-normalizes each vector, computes an equal-weight
mean, and unit-normalizes that mean. Opposite vectors whose mean is zero are an error, not an
invented speaker identity. The functions neither retain nor persist their inputs.

`AdaptiveScoreNormalizer` is optional symmetric adaptive cohort normalization (AS-norm).
It copies a cohort of equal-dimensional vectors. For each side of a trial, take its highest
`topCount` cosine scores against the cohort (default 100, capped at cohort size, minimum two).
Compute their mean and population standard deviation. The returned score is
`0.5 * ((cosine - mean1) / std1 + (cosine - mean2) / std2)`.
Standard deviations are floored at 1e-6 by default. Scores are unbounded, not probabilities.
Use a disjoint, consented development cohort; evaluation identities must not enter the cohort.
The library does not select, fetch or persist one.

`LogisticScoreCalibration` fits an affine sigmoid to labeled **development scores**, not
speech-model weights. Labels indicate same-speaker trials. The objective is mean binary
cross-entropy plus half the specified L2 penalty times the sum of squared coefficients.
Both coefficients are penalized; the default penalty is 1e-4. Scores are standardized with
population standard deviation floored at 1e-6. Zero-initialized, fixed-step gradient descent
runs 100 iterations by default, with step `1 / (0.5 + penalty)`; there is no randomness.
Both classes must occur. Output probabilities reflect the development set's empirical class
prior, not a deployment population prior. Changing trial balance changes their meaning.
Choose a deployment threshold on held-out data, never on the reported evaluation trials.

## In-memory registry

`InMemorySpeakerRegistry` implements the existing `SpeakerRegistry` contract. It borrows a
`SpeakerEmbedder`, serializes its own extraction calls, and stores only unit-normalized
averages keyed by case-sensitive IDs. The caller must coordinate any embedder use outside
this registry. Enrollment replaces an ID atomically after all utterances succeed; the overload
accepting multiple utterances uses the arithmetic above. A failed or cancelled extraction leaves
the previous enrollment intact. Very short or silent audio remains the embedder's responsibility.

The constructor requires an explicit, inclusive acceptance threshold; no universal threshold is
claimed. Identification returns the highest score, breaking exact ties by ordinal ID. Below the
threshold it returns null ID with the best score. An empty registry returns null ID and negative
infinity. Verification scores a pair without modifying enrollment. Raw scoring is cosine unless
an optional AS-norm cohort is supplied. Calibration is separate and is never silently applied.
`Remove` clears an enrollment and returns whether it existed. `Dispose` clears all enrollments,
but does not dispose the borrowed embedder. Clearing managed arrays is best effort: it is not
a memory-erasure guarantee against runtime copies, crash dumps or a hostile host.

There is no embedding serialization, file storage, network enrollment or process-wide model
selection in this implementation. Hosts can share one registry instance in their process;
convenience static integration is separate.

**Privacy:** speaker embeddings and enrollment audio are biometric data. Obtain appropriate
consent and determine applicable local-law requirements, including Illinois BIPA and GDPR
Article 9 where relevant. Avoid logging embeddings, restrict access to process memory, and
remove enrollments when no longer needed. The library never persists embeddings or audio.
Tests use algebraic vectors, not recordings, generated voices or biometric identities.

## Checkpoint inspection

The development converter can fetch and SHA-256 verify four publisher checkpoints:

```sh
dotnet run --project tools/OneSpeech.Convert -- speaker-fetch /caller/selected/model-cache
dotnet run --project tools/OneSpeech.Convert -- speaker-inspect /caller/selected/checkpoint
```

The fetch command verifies existing files too; a mismatch fails rather than overwriting an
existing checkpoint. Downloads are bounded to 128 MiB, verified before rename, and temporary
files are removed on failure. Do not run concurrent fetches against the same directory.
Inspection prints only tensor names and dimensions, not a runtime graph or inferred architecture.
These commands do not register a model or select a default embedder.

Inspection supports a deliberately restricted tensor-only ZIP/pickle data language. It never
executes Python or invokes globals. Only inert ordered-map, storage and tensor-view constructors
are accepted; all other global names, reductions and object state are refused. Float32, float64,
float16 and int64 storage can be converted to float32 arrays, including bounded noncontiguous
views. Optional ordered-map metadata is parsed under the same bounds and discarded. No archive
entry is extracted to disk. All index text is strict UTF-8.

Limits: 256 MiB archive, expanded entries, and output arrays (each budget independently);
8 MiB central directory and pickle index; 20,000 ZIP entries; rank at most 8;
dimensions at most 10,000,000; no negative strides or views outside storage; two million parser
operations and aggregate collection slots; 200,000 stack and memo entries; 65,536-byte strings.
Multi-disk and ZIP64-only directory records are not supported. Invalid data throws
`InvalidDataException`; ordinary stream I/O failures remain I/O errors. Inspection is a developer
tool, not the shipped model-loading API. No embedding model implementation or parity claim is
provided by checkpoint inspection.

## CAM++ inference

`OneSpeech.Embedders.CampPlusEmbedder` implements `SpeakerEmbedder` over a locally converted
OneModel graph. Construct it with an explicit graph path, truthful checkpoint `ModelInfo`, and
embedding dimension (512 for the English VoxCeleb checkpoint; 192 for the Chinese common
checkpoint). The default is one CPU thread. It never downloads, trains, saves embeddings, or
selects a default model. Input is mono 16 kHz float PCM, 45 ms through 60 s; destination storage belongs to the caller.
Inference calls and disposal are serialized per instance. Cancellation is checked during feature
extraction and before graph inference; OneModel's synchronous graph call itself is not interruptible.

`SpeakerFbank` computes 80 natural-log mel energies on complete 400-sample windows at 160-sample
hops. It uses unscaled float PCM, removes each window's DC component, applies 0.97 preemphasis
(with the first sample repeated), a symmetric Povey window, and a zero-padded 512-point power FFT.
Mel triangles span 20 Hz through 8 kHz on the mel scale. Energies are floored at float machine
epsilon before taking natural logs. Each feature dimension has its utterance mean removed.
Output is planar `[mel, frame]`, not interleaved frames. These settings were observed by a
pass-through capture at the documented public fbank API invoked by ModelScope 1.40.2. The wrapper
returned all values unchanged; no implementation source was inspected. The extractor is deterministic,
requires finite samples, and accepts 25 ms through 60 s. A partial final window is not padded.

The converter implements Wang et al. (2023), Sections 2.1–2.4: frequency-downsampling residual
FCM, densely connected TDNN blocks of 12/24/16 layers with growth 32 and dilations 1/2/2, and
context-aware masks combining global and consecutive 100-frame segment means. An incomplete last
segment is divided by its actual frame count. All nonlinear normalization stages use BN then ReLU.
Statistics pooling concatenates the mean and unbiased standard deviation before the final projection
and non-affine batch normalization. Architecture details left unstated by the paper were settled by
black-box numerical probes with identical input features. All neural operators run in OneModel;
there is no native inference dependency.

```sh
dotnet run --project tools/OneSpeech.Convert -- speaker-convert-camp checkpoint.bin camp.onemodel
```

The checkpoint must already have been verified against its publisher pin. Conversion does not
execute pickle objects or load Python. The graph and weights stay outside the repository.
No corpus-quality or default-selection claim follows from successful conversion alone.

## Verification slice

`tools/scripts/generate-speaker-trials.sh TEST_CLEAN TEST_OTHER trials.tsv 1729` constructs ID-only
trials from separately fetched LibriSpeech test-clean/test-other data. Sort IDs ordinally, group by
speaker, and retain the first 20 utterances per speaker. A seeded .NET 10 PRNG selects speakers
uniformly and utterances uniformly within speakers; reject identical utterances and duplicate unordered
pairs. Same-speaker and different-speaker classes contain 10,000 unique pairs each. Trials are finally
sorted by their two IDs. No audio or embeddings are included in the trial file. This is an open-data
verification slice, not VoxCeleb1-O and not an estimate of a deployment prior.

`speaker-score trials.tsv OUR_EMBEDDING_DIR REFERENCE_EMBEDDING_DIR result.json` applies the same
cosine scorer to both embeddings, grouping equal scores at ROC thresholds. EER linearly interpolates
at the false-accept/miss crossing. Minimum DCF uses target prior 0.01, miss and false-alarm costs 1,
and normalization by the always-reject cost (0.01); it includes the always-reject operating point.
The paired bootstrap uses 1,000 resamples, seed 1729, independently sampling trials with replacement
within each same/different class while keeping our/reference scores paired. Percentile endpoints use
linear quantiles at 0.025 and 0.975. Confidence intervals describe the score difference conditional on
the fixed utterance/speaker set, not uncertainty over independent unseen speakers; trials sharing
speakers are not independent samples of a speaker population. No normalization or score calibration
is applied to these evaluation rows.

The English VoxCeleb checkpoint at revision `032b8131a7ad812f87061955ca974c99060c5a03` was checked
against ModelScope 1.40.2 (torch/torchaudio 2.11.0 CPU, float32) on the 50 IDs in
`eval/manifests/librispeech-speaker-parity50.txt`. Feature maximum absolute error was 0.000597 and
mean absolute error 0.00000624; the documented float32 feature tolerance is max 0.001 on these
probes. Embedding minimum/mean cosine were 0.999999999957/0.999999999985, above the 0.999 target.
This is numeric parity on a fixed probe set, not a speaker-verification quality claim. The converted
English graph is 29,414,032 bytes, SHA-256
`f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c`.
