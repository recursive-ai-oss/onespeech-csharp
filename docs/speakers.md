# Speaker identity and scores

`SpeakerEmbedder` is the synchronous CPU extraction role. Its model metadata identifies the
checkpoint and licence; `Dimension` fixes output length and `SampleRate` fixes mono input Hz.
`Embed` writes into caller-owned storage and accepts cancellation. Input must be nonempty,
mono, and already resampled. Implementations return finite, nonzero vectors and reject unsupported
input rather than silently changing channels. Callers dispose model resources after consumers stop.
No checkpoint implementation is included yet.

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
