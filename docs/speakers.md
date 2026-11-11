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

There is no embedding serialization, file storage or network enrollment. Hosts can share one
registry instance in their process; the separate `Speech` helpers manage a process-wide registry
per supported CAM++ model, with calibrated identification thresholds described below.

**Privacy:** speaker embeddings and enrollment audio are biometric data. Obtain appropriate
consent and determine applicable local-law requirements, including Illinois BIPA and GDPR
Article 9 where relevant. Avoid logging embeddings, restrict access to process memory, and
remove enrollments when no longer needed. The library never persists embeddings or audio.
Arithmetic tests use algebraic vectors. Speaker integration tests use public LibriSpeech speech only; embeddings are never committed.

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
FFT twiddles are cached at construction-independent fixed sizes. Mel accumulation visits only each triangle's nonzero support, in the original bin order; there is no SIMD reduction or change of rounding order. The optimized path is bit-identical to the dense arithmetic on five fixed LibriSpeech probes and the synthetic test inputs. All power bins, including zero-weight bins, are checked for overflow.
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

On the committed 20,000-trial LibriSpeech verification slice, both this graph and the pinned
ModelScope reference scored **0.75% EER** and **0.0379 normalized minDCF**. The paired bootstrap
95% intervals for our-minus-reference differences were [0,0] percentage points and [0,0], respectively:
the relevant ROC operating points were identical. These are conditional trial-resampling intervals,
not confidence intervals over a new population of speakers. The same feature extractor was also
checked on deterministic silence, tone, impulse, DC-offset, clipped, seeded-noise, exact-window and
two-window probes; maximum absolute error across them was 0.000851 (within the 0.001 tolerance).

These CPU/float32 checks used Linux x64, an AMD Ryzen 9 7950X3D (32 logical CPUs, AVX2 and AVX-512
available), eight inference threads, .NET SDK 10.0.403/runtime 10.0.14, OneModel 1.0.0 and
OneAudio 1.7.2. No GPU was used. Oracle dependencies and wheel hashes are pinned in the speaker
oracle's `uv.lock`; model and identifier pins are given above. Conversion is deterministic: its
recorded graph SHA-256 was reproduced from the same checkpoint independently.

## ECAPA-TDNN

`EcapaTdnnEmbedder` accepts the converted VoxCeleb checkpoint and truthful `ModelInfo`, returning
192 raw embedding values at mono 16 kHz. Calls are serialized; inputs must be 40 ms through 60 s.
The minimum ensures reflected temporal convolution pads are smaller than the frame dimension.
The graph follows Desplanques et al. (2020), Sections 3.1–3.3 and the pinned checkpoint config:
1024-channel TDNN, three scale-8 SE-Res2Blocks (dilations 2/3/4), multi-layer aggregation,
channel-dependent attentive statistics pooling and the final 192-dimensional projection. Convolution
uses reflected padding; TDNN activation is Conv → ReLU → BN. The first Res2 split bypasses its
convolution; subsequent splits add the preceding processed split. Global and weighted pooling
standard deviations use population statistics with a 1e-12 variance floor. These details were
confirmed against numeric black-box outputs, not implementation source.

`EcapaFbank` is a distinct public feature profile, not the CAM++ filterbank: centered zero-padded
400-sample periodic Hamming windows, 160-sample hops, 400-point power FFT, no preemphasis/DC removal
or amplitude scaling. There are 80 symmetric Hz-domain triangles with centers evenly spaced in
HTK mel from 0 through 8 kHz; each triangle's half-width is its center's distance from the preceding
center. Energies are floored at 1e-10, converted by 10 log10, clipped at the utterance maximum minus
80 dB, and mean-normalized by feature dimension. `ComputeRaw` exposes the decibel energies before
mean subtraction. Numeric public feature probes settle this profile; the model paper's MFCC input
description does not replace its released checkpoint's Fbank configuration.

The 50 committed LibriSpeech probe IDs, using SpeechBrain 1.1.2 with torch/torchaudio 2.11.0 CPU,
produce max/mean normalized-feature absolute errors 0.001542/0.00002079 dB, and minimum/mean embedding
cosine 0.999999999983/0.999999999990. Feature tolerance is max 0.002 dB (about 0.000461 natural-log
energy units); embedding target is cosine >=0.999. The mixed-radix 400-point transform uses managed
16-point FFTs and 25-way combination, not a native frontend. Feature-only probes also cover
silence, tone, DC-offset, impulse, seeded noise and sub-window audio. The feature extractor accepts
1 sample through 60 s; the model's stricter 40 ms limit still applies.

```sh
dotnet run --project tools/OneSpeech.Convert -- speaker-convert-ecapa embedding_model.ckpt ecapa.onemodel
```

The converted graph is 83,294,800 bytes, SHA-256
`f107ead463783cab70914fd0470cc6b42e2ec74b7a717bb42c1360879076c669`. As with CAM++, its catalog
release URI is reserved/unpublished; the model store converts verified publisher files locally. ECAPA is available by name; CAM++ English is the speaker default.

On the same 20,000-trial LibriSpeech slice, ECAPA and its pinned reference both score **1.10% EER**
and **0.0548 normalized minDCF**, using the same raw cosine scorer, prior/costs and conditional
paired-trial bootstrap described above. Our-minus-reference intervals are [0,0] for both metrics.
The ECAPA oracle uses the publisher's exact `/resolve/<commit>` URL as its source and pretrained
path, because the released `from_hparams` interface does not accept a `revision` constructor keyword.
Auxiliary classifier/global-normalization files loaded by the oracle are recorded in notices but are
not part of the converted graph; embedding normalization is explicitly disabled on the reference call.

## CAM++ Chinese common checkpoint

The same converter/frontend and `CampPlusEmbedder` (dimension 192) support
`iic/speech_campplus_sv_zh-cn_16k-common` revision `013c18884dffb82846cf472dd06dd4a08d97e2cd`.
On the same 50 IDs, feature max/mean errors are 0.000597/0.00000624 and embedding minimum/mean
cosines are 0.999999999966/0.999999999984. This meets the numeric parity target.
The converted graph is 28,098,192 bytes, SHA-256
`4b2f0fef5ec8cb855b53c6b5abae5dd4cd72c215137c52d32f5c477e4a792f49`, registered under the same
unpublished-artifact convention. No default selection is implied.

**Information only:** this checkpoint is trained on Chinese speech, while these trials contain
English read speech. Both our graph and the reference score 2.87% EER and 0.1876 normalized minDCF
on the shared 20,000-trial English slice, with paired difference intervals [0,0]. The same prior,
costs and conditional trial bootstrap apply. This is an implementation comparison and not a
language-appropriate accuracy bar or a new-speaker population confidence interval.

## ERes2NetV2 Chinese common checkpoint

`ERes2NetV2Embedder` (192 dimensions, 200 ms through 60 s of mono 16 kHz audio) runs
`iic/speech_eres2netv2_sv_zh-cn_16k-common` revision `1cf80d41fb3435bd3d8df185b5c423333b2db42a`
(Apache-2.0, tier 1). It uses the CAM++ 80-bin filterbank and follows Chen et al. (2024, arXiv
2406.02167) §2: a 3x3 stem, four stages of 3/4/6/3 bottleneck-like blocks (1x1 reduce, two 3x3 splits,
1x1 expand), attentional feature fusion between the splits in stages 3 and 4 and between the
down-sampled stage 3 and stage 4, statistics pooling over time and one linear layer. The paper leaves
several details open; they were settled by comparing stage outputs of the released checkpoint:
the stride sits in each block's first 1x1 convolution and its shortcut; the fusion is
`x*(1+a) + y*(1-a)` with `a = tanh(att)`; the in-block fusion is followed by ReLU and the stage 3/4
fusion is not; a block's activation after the residual sum is capped at 20; pooling takes the mean and
unbiased standard deviation over time of the channel-by-frequency map.

On the 50 fixed IDs, embedding cosine against ModelScope 1.40.2 (CPU, fp32) has minimum
0.9999977 and mean 0.99999995 (over all 1,490 trial utterances: minimum 0.9999977, none below
0.9999); the target is >=0.999. Features are the CAM++ features (max abs error 0.000597, mean
0.00000623). The converted graph is 71,723,456 bytes, SHA-256
`ea17d67d75abcaa3d180a6f8b81559c58cbd701a6235a38f27a9f2aa2fca4a7a`, reproduced by a second
conversion; as with the other checkpoints its release URI is reserved and the model store converts the
verified publisher file locally. It is available by name and is not a default.

**Information only** (Chinese-trained model, English trials): on the shared 20,000-trial slice both
ours and the reference score 1.33% EER and 0.0658 normalized minDCF, paired difference intervals
[0, 0] (same prior, costs and conditional bootstrap as above).

The released pipeline fails to start from the published `configuration.json`, which omits three
architecture keys it requires; the oracle runs from a local copy of the pinned files with
`embed_dim`, `baseWidth` and `scale`/`expansion` added (`tools/oracles/speaker-modelscope/eres_config_loop.py`).

## Convenience identification operating points

The file helpers use raw cosine without AS-norm or a fitted affine map. Threshold fitting uses
`eval/manifests/librispeech-speaker-trials.tsv` (SHA-256
`8a7e63167cd625bd0633a29bd5f517ceb37e9a32fccec062328a52fd44aa43ae`, seed 1729).
Sort its 73 distinct speaker IDs ordinally; the first floor(73/3) = 24 identities are development,
the remaining 49 are holdout. Retain a trial only when both identities belong to the same partition;
discard cross-partition trials. This is a partition of the existing LibriSpeech test-clean/test-other
trial manifest, not the official LibriSpeech dev split and not a new independent benchmark.
Neither holdout scores nor query examples select the threshold.

Development IDs: 1089, 1188, 121, 1221, 1284, 1320, 1580, 1688, 1995, 1998, 2033, 2094,
2300, 237, 2414, 260, 2609, 2830, 2961, 3005, 3080, 3331, 3528, 3538.
There are 3,296 positive and 1,036 negative development trials. Choose the observed score minimizing
normalized DCF = miss rate + 99 × false-accept rate (target prior .01, unit miss/false-accept costs).
Acceptance is inclusive; cost ties prefer the larger threshold. Also report interpolated EER and
the observed threshold nearest equal false-reject/false-accept rates (stricter on ties).
`OneSpeech.Convert speaker-calibrate <manifest> <embedding-directory>` reproduces the selection
from the existing parity-checked embeddings without changing any model weight.

| Model | Dev EER % | EER operating point | Dev minDCF | Identification threshold | Holdout miss % | Holdout false accept % | Holdout DCF |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM++ English | 0.6068 | 0.41554184019080986 | 0.01638 | **0.4929986257373565** | 3.1921 | 0.02227 | 0.05397 |
| CAM++ Chinese | 2.7002 | 0.5890692253248851 | 0.16292 | **0.7229075438038025** | 15.5131 | 0.06682 | 0.22128 |

Holdout contains 11,194 retained trials. These are conditional trial rates, not population
confidence intervals or an authentication guarantee. Chinese CAM++ is measured on English read
speech here; this does **not** establish a Chinese-speaking population operating point. Open-set
identification with many enrolled names and cluster-average embeddings has different error rates
from pair verification. Deployments must validate their own population, recording conditions,
consent and risk policy using the low-level registry's explicit threshold. The convenience
operating point is a conservative read-speech default, not a liveness check or anti-spoofing system.
