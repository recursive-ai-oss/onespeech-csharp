# Diarization

Speech recognition for .NET. Speaker labels are document-local, not biometric identities. No audio or embedding is persisted by diarization helpers.

## Word assignment

`TranscriptDiarization.Merge` is shared by diarizers. Input `ActiveTurn` values wrap original-clock turns and optional relative activity in [0,1] (default 1, not a probability). Turns require a named, assigned speaker, nonnegative start and positive duration. Null times are never filled in.

Intervals are half-open. A sweep partitions the turns at every endpoint. Where speakers are concurrent, the highest-activity turn wins; equal activity breaks by ordinal speaker ID. Covered duration in these partitions is summed per speaker for each word. The speaker covering the greatest duration wins; equal durations again break by ordinal ID. Uncovered portions do not vote. A positive-duration word can therefore be assigned even if most of its interval is silence. No coverage, null endpoints, reversed endpoints, and zero-duration words yield `SpeakerLabel.Unassigned`.

`Word.Overlap` is true only when a positive-duration part of the word intersects simultaneous turns from **different** speakers. Duplicate or overlapping turns from the same speaker do not count as overlap. Touching endpoints do not overlap. This flag is independent of which speaker wins. Merging replaces prior speaker/overlap assignments; it preserves word text, timing, confidence and timing method, document text, language and model provenance. Segment words are assigned by the same rule. The input document remains unchanged.

Original turns are ordered by start and ordinal speaker ID. Each assigned root word is attached once to the turn of its chosen speaker having greatest temporal intersection, breaking ties by earliest start. Turn text is the exact concatenation of attached word text, without inserted spacing. Segment-only words are assigned in their segments but are not added to the root word list or turn text.

`TranscriptDiarization.Exclusive` provides a separate, exclusive caption timeline: the highest-activity speaker at each instant, with the same tie rule, coalescing adjacent intervals for the same speaker. It leaves gaps uncovered and produces empty turn text/words; a caller must place captions explicitly rather than infer word timing from this view.

## Separated-party stereo

`ChannelDiarizer` uses channel labels `channel-1` and `channel-2`; it does not infer identities or distinguish multiple people on one channel. Auto-detection is deliberately conservative:

* exactly two channels;
* at least three 20 ms frames dominated by the left channel and at least three dominated by the right;
* dominated means energy at least 100 times the other channel (20 dB), above mean-square energy 1e-8;
* absolute normalized waveform dot product over the whole recording at most 0.3.

The final short frame is included. These are initial heuristic values, **not** thresholds fitted or validated on a speech development set. Silence, mono, dual-mono and one-sided conversations are inconclusive, not evidence of separation. Auto-detection returns false for them. This rule cannot prove separation: uncorrelated stereo music or alternating noises may pass. Known separated calls can set `forceSeparated: true`; known mixed audio should use a voice-clustering diarizer instead. Failure is a `CapabilityNotSupportedException`, never a silent downmix fallback.

For each channel, VAD runs on bounded, independently analyzed blocks of at most 30 seconds, retaining original-clock times. VAD state resets at block boundaries; this is not continuous-state VAD equivalence. Simultaneous channel speech is retained. Channel output metadata grows with turns, but temporary PCM does not grow with duration. The caller's in-memory input remains borrowed.

## Pipeline

The five-argument `VadTranscriptionPipeline` constructor accepts a borrowed `Diarizer` after the existing chunk-duration and progress arguments. The four-argument constructor is unchanged. In-memory transcription runs recognition on the usual downmix and diarization on the **original channels**, then merges words. Segment enumeration computes whole-recording diarization before emitting any final segments, so assignments are never revised after emission.

`ClusteringDiarizer` and `ChannelDiarizer` implement `BlockDiarizer`: consecutive PCM blocks share one recording clock, and returned IDs are global to that recording. File inputs make two bounded decoding passes: diarization first, recognition second. Encoded streams use the existing temporary seekable spool and rewind it between passes; the spool is deleted when processing ends. No recording waveform is retained. Progress remains recognition audio-time progress, not wall time or a sum over both passes. Callers must not change a file while it is being processed.

A custom plain `Diarizer` still works in memory, but file inputs require `BlockDiarizer`; otherwise they throw `CapabilityNotSupportedException` naming that requirement. BlockDiarizer declares contributing `Models`, which the pipeline adds to document provenance, deduplicated by ID and revision. Both channel and clustering implementations declare their VAD/embedding models. Omitting the diarizer preserves existing bounded file transcription.

## Voice clustering and overlap limitation

`ClusteringDiarizer` is the tier-1 voice-clustering implementation, using caller-supplied VAD and speaker-embedding roles. It **does not detect overlapped speech**: each speech interval gets only one speaker at a time. The overlap-aware [Nemotron 3 Diarization](nemotron-diarization.md) upgrade is an explicit tier-2 option, never selected by these helpers; the pyannote segmentation slot remains separate. Channel diarization is a distinct exception because separated channels supply activity independently. The component itself always uses the caller's embedder. The `Speech.Transcribe` helpers select CAM++ Chinese for a requested or detected Chinese language (`zh`, `zh-*`); otherwise they select CAM++ English. An explicit language takes precedence over detection. Recognition runs before the whole-recording diarization pass so this policy also applies without a language hint. Encoded streams are spooled once, with bounded PCM on both passes. This is a language-based starting choice, not a quality-validated default for every language.

For an overlap-aware file transcript, keep your recognizer and VAD and replace only the diarizer:

```csharp
using var overlap = await OneSpeech.Diarization.NemotronDiarizer.LoadAsync();
var transcript = new OneSpeech.Pipeline.VadTranscriptionPipeline(recognizer, vad, null, null, overlap).Transcribe(path, language: null, CancellationToken.None);
```

The pipeline retains the higher-activity speaker in simultaneous intervals, flags overlapping words, and leaves gaps Unassigned. This does not separate simultaneous voices or recover words the recognizer missed. The named load fetches the publisher's pinned checkpoint, keeps OpenMDW agreement/origin notices in `ModelInfo`, and converts only into the local cache.

### Windows and clock

Input is consecutive 16 kHz PCM with one consistent channel count (1–32). Channels are averaged for voice clustering. VAD runs in fixed 30-second horizons, independent of caller block boundaries; state resets at each horizon. VAD regions are sorted and overlapping/touching regions unioned. Segments shorter than 100 ms are discarded. Following Landini et al., arXiv:2012.14952, §3.2, defaults are 1.5-second embedding windows and a 0.25-second stride; partial trailing windows of at least 100 ms are included. Models are borrowed, and each finite, nonzero embedding is length-normalized in double-precision accumulation before clustering. Embedding failures propagate; they are not replaced by arbitrary labels.

The boundary between neighboring window assignments is the midpoint between their centers. First/last assignments extend to the **VAD region** endpoints. Silence gaps remain gaps. Adjacent same-speaker assignments are coalesced. Labels `speaker-1`, `speaker-2`, ... follow first occurrence, are document-local, and are not enrolled identities. These are window-level speaker boundaries, not refined word or phonetic timing.

### Spectral clustering and AHC

The default `ClusteringMethod.Spectral` follows Park et al., *Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap*, arXiv:2003.02405v1, §III, Eq.4–15:

1. Raw cosine affinities; row-wise top-p binarization (stable index ties), including the diagonal when ranked there.
2. Symmetrization `(A + Aᵀ)/2`, then unnormalized Laplacian `D − A`.
3. Symmetric eigendecomposition and ascending eigenvalues. The largest allowed eigengap estimates the speaker count. Select p minimizing `p × (largestEigenvalue + 1e-10) / max(gap,1e-10)`.
4. Cluster rows of the smallest-k eigenvector matrix by Euclidean Lloyd k-means, with deterministic farthest-first seeds (first seed index 0), stable ties and empty-cluster repair. At most 100 iterations.

Two bounded-work deviations from the paper are explicit: dense clustering uses up to `MaxLandmarks` uniformly spaced windows over the complete sequence (default 128, allowed 16–256); p is searched at 2,4,8,... plus `max(2,floor(L/4))`, rather than every integer up to L/4. All nonsampled windows are assigned to the maximum-cosine landmark-cluster centroid; landmark labels are retained, including explicit-count splits of indistinguishable vectors. Rare/short speakers absent from the sample can therefore be missed. This is not a reference-equivalence claim.

The symmetric eigensolver uses cyclic Jacobi rotations, at most 60 sweeps, stopping when maximum off-diagonal magnitude is ≤1e-10 times the largest diagonal magnitude (at least 1). Numerical nonconvergence uses AHC instead. When all landmark cosines are within 1e-6 of one, there is no geometric separation evidence; AHC avoids creating speakers from arbitrary graph tie-breaking.

`ClusteringMethod.Agglomerative` is the alternative and the spectral numerical fallback: average-link cosine similarities, stable pair-order ties, merging to the supplied exact count or stopping below the configurable cosine threshold while honoring min/max. This uses the cosine AHC initialization described in Landini §3.2, **not** PLDA. The generic threshold 0.65 is an exposed starting value, not a universal model calibration; use the model-specific measured profile below. No embedding-model weights are changed.

`NumSpeakers` is exact for nonempty speech; `MinSpeakers`/`MaxSpeakers` constrain estimates. Default range is 1–8, supported ceiling 32. Exact counts must lie within min/max. If speech produces fewer windows than the required count, the request is refused with `CapabilityNotSupportedException`; silence produces no turns, not fictitious speakers. Invalid parameter combinations are rejected. Cancellation is checked per PCM block, embedding window, eigen sweep, clustering iteration and output turn.

### Memory and resegmentation

Only the embedding sequence and interval metadata grow with file duration. Temporary PCM is bounded by 30-second buffers (plus the decoder's bounded state); dense clustering is O(L²), capped at L=256. Retained sequence storage is O(ND+N), with float embeddings requiring 4ND bytes plus window/interval/collection overhead, labels and O(KD) centroids. This is **not** constant total memory independent of duration. Default `MaxWindows=100000` (configurable up to 1000000) refuses further windows explicitly instead of growing without a limit. At the default stride, continuous speech produces approximately 14400 windows/hour; dimension 192 needs approximately 11 MB/hour of vector payload, dimension 256 approximately 15 MB/hour, before metadata. Caller-owned in-memory PCM is outside this bound.

An optional `EmbeddingResegmenter` receives borrowed windows/initial labels synchronously and returns a label per window before turn construction. The output must use only initial labels and respect count hints. The low-level constructor installs none by default; file helpers use the cosine-centroid profile below. In particular **VBx is deferred**. Landini §§2.1–2.3 requires a compatible pretrained PLDA mean, whitening/LDA transform and between-speaker variances. None is fitted or substituted here. Embeddings are never written to disk or retained by the diarizer after the call.

### Development parameters and evaluation

CAM++ English (512 dimensions, float32) was fitted on **AMI Full-corpus-ASR dev ES2011a/ES2011b only**, with the original Mix-Headset signal and word-derived references described in the fetch scripts. Window defaults remain the paper-based 1.5 s / 0.25 s; count range is 1–8, with no oracle speaker count. The grid uses 64/128/256 landmarks and AHC thresholds −0.2, −0.1, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85. The initial high-threshold grid was extended using development results only. Selection minimizes pooled speaker-time DER, ties by reference-speaker-macro JER then ordinal profile ID. Embeddings and VAD outputs are replayed from RAM after input SHA checks while varying clustering parameters; no embeddings are saved or model weights altered.

| CAM++ en profile | Method | Landmarks | AHC threshold | Pooled dev DER / JER |
|---|---|---:|---:|---:|
| S128, selected primary | Spectral/NME |128|0.65 numerical fallback only|33.3520% / 42.6786%|
| A64-0.15, fitted alternative | Average-link AHC |64|0.15|57.0071% / 72.1743%|

These are model-specific recipes, not a claim that the generic options are optimal for every embedder. S128 coincides with the existing spectral defaults. To use the fitted AHC alternative, explicitly set `Method=Agglomerative`, `MaxLandmarks=64`, `AgglomerativeThreshold=0.15`; retain 1.5 s / 0.25 s windows and min/max 1/8. All selections were frozen before reading holdout scores.

Channel-layout calibration used AMI dev ES2011a/ES2011b Headset-0/1 only. Ten correlated proxies total (five per meeting) cover separated microphones, 1% statistical leakage, 25%/50% leakage negatives, and dual mono. Leakage is computed on frame energy/cross-energy statistics; no new speech waveform is rendered. Searching 10/15/20/25/30 dB and normalized-dot limits 0.2/0.3/0.4/0.5/0.6 retained **20 dB / 0.3**, with 4 true positives, 6 true negatives and no errors. Ties prefer the smallest change from the initial constants. The fixed three-frame and energy-floor rules are unchanged. This is not independent telephony accuracy or evidence that the heuristic proves separation.

### Initial holdout floors

The embedder is ModelScope `iic/speech_campplus_sv_en_voxceleb_16k` revision `032b8131a7ad812f87061955ca974c99060c5a03`; converted graph SHA-256 `f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c`. Linux x64, AMD Ryzen 9 7950X3D (32 logical CPUs, AVX2/FMA/AVX-512 available), .NET SDK 10.0.403/runtime 10.0.14, CPU/float32, eight inference threads, OneModel 1.0.0 and OneAudio 1.7.2; no GPU or speed measurement. Silero v6.2.3 uses threshold 0.5, negative 0.35, minimum speech 250 ms, minimum silence 100 ms, pad 30 ms, with the documented independent 30 s horizons. No collar; all overlap scored; no UEM exclusions. Figures are **first-measurement floors**, not reference parity or an absolute quality pass.

| Held-out meeting | S128 DER / JER | A64-0.15 DER / JER |
|---|---:|---:|
| AMI test ES2004a, Mix-Headset |33.9706% / 38.1994%|36.2530% / 39.3777%|
| AMI test ES2004b, Mix-Headset |27.0682% / 29.1370%|28.8334% / 29.0338%|
| AliMeeting Eval R8001_M8004_MS801, far channel 1 |66.1638% / 64.4226%|67.9955% / 65.5995%|
| AliMeeting Eval R8003_M8001_MS801, far channel 1 |45.3672% / 51.7859%|47.6219% / 53.7021%|

Pooled S128: AMI DER 29.0875%, JER 33.6682%; AliMeeting DER 55.2437%, JER 58.1043%. Descriptive 95% bootstrap intervals over the **two meetings** (1,000 resamples, seed 1729) are respectively DER [27.0682,33.9706]%, JER [29.1370,38.1994]%, and DER [45.3672,66.1638]%, JER [51.7859,64.4226]%. They do not establish generalization to a new speaker/meeting population. Individual-meeting intervals are unavailable. Missed speaker-time dominates these rows; this measurement does not separate VAD misses from the known inability to represent concurrent speakers.

### Language-appropriate evaluation and published bundle comparison

CAM++ Chinese (`iic/speech_campplus_sv_zh-cn_16k-common`, revision `013c18884dffb82846cf472dd06dd4a08d97e2cd`, converted SHA-256 `4b2f0fef5ec8cb855b53c6b5abae5dd4cd72c215137c52d32f5c477e4a792f49`) improves the Mandarin rows with the **same frozen settings**, without fitting on AliMeeting Eval:

| AliMeeting Eval meeting | Chinese S128 DER / JER | Chinese A64-0.15 DER / JER |
|---|---:|---:|
| R8001_M8004_MS801 |62.3775% / 59.4958%|72.9761% / 82.1277%|
| R8003_M8001_MS801 |39.6415% / 42.5590%|48.2732% / 59.3365%|

A released ModelScope 1.40.2 public `speaker-diarization` pipeline was run black-box with its published bundle `iic/speech_campplus_speaker-diarization_common` at `0b056318fde24067871436d60d020fa01f3096b4`. Unlike passing a speaker-verification model directly to a diarization task, this is the documented bundle. It uses its own FSMN VAD, Chinese CAM++ and speaker-change locator, with default settings and no speaker-count oracle. Only auxiliary configuration paths were changed to verified local copies; network access was disabled. Pins are in `tools/oracles/diarization-modelscope/diarization-bundle.tsv`; its driver is `bundle.py` and dependency lock is under `tools/oracles/diarization-modelscope`. CPU float32, four threads/affinity to four logical CPUs, Python 3.13.17 / Torch 2.11.0 CPU, same host and scoring conventions as above. These are **different systems**, not logit-parity measurements.

| Meeting | Our S128 DER / JER (English for AMI, Chinese for AliMeeting) | ModelScope bundle DER / JER |
|---|---:|---:|
| ES2004a |33.9706% / 38.1994%|26.5359% / 29.6337%|
| ES2004b |27.0682% / 29.1370%|18.1017% / 19.2563%|
| R8001_M8004_MS801 |62.3775% / 59.4958%|34.5196% / 34.0168%|
| R8003_M8001_MS801 |39.6415% / 42.5590%|26.8887% / 28.5758%|

Every row misses a 0.5-point difference bound. A single meeting cannot supply an utterance/file bootstrap interval; these fixed slices do not prove population quality. Missed speech is the largest component in all our S128 rows. Replacing the embedder alone does not address the VAD/overlap weakness; no cause beyond this loss breakdown is established here.

A real 300-second AMI ES2004a excerpt through `Speech.Transcribe` with Whisper small and the original 0.5-threshold / 30-ms-padding activity settings produced 344 words, WER **16.0584%** against chronological timed AMI words (`en/1`, 411 reference words), 28 Unassigned, zero Overlap and zero untimed words. Under an optimal one-to-one speaker mapping, 290/344 (**84.3023%**) predicted words had the correct reference speaker; 290/335 (**86.5672%**) among words with reference-time coverage. Each predicted word's reference speaker is the one covering most of its interval, with ordinal ties. This is temporal speaker assignment, not lexical correctness; Unassigned, untimed and no-reference words count against the all-word fraction. These are informational excerpt results, not a long-form quality bar. `onespeech-eval diarize speech-excerpt DATA_ROOT ES2004a OUTPUT openai/whisper-small` writes the transcript JSON and scoring summary. Holdout references were never used to choose settings.

See [evaluation](eval.md) for no-collar, overlap-inclusive DER/JER and RTTM input.

## Diarization-specific speech activity

The file transcription helpers use **separate** Silero instances for ASR chunking and diarization. ASR and live chunking retain the published defaults (activation 0.5, deactivation 0.35, minimum speech 250 ms, minimum silence 100 ms, padding 30 ms). The diarization instance uses activation **0.2**, deactivation **0.2−0.15** (float32, approximately 0.05), minimum speech **250 ms**, minimum silence **250 ms**, padding **150 ms**. Independent 30-second horizons and the 100-ms minimum embedding segment remain unchanged. Low-level `ClusteringDiarizer` continues to use the caller-supplied detector; it does not override its options. No ASR VAD behavior or model weights change.

These activity settings were selected only on AMI development ES2011a/ES2011b. AliMeeting Train/dev was not present and AliMeeting Eval was not used for fitting. The activity grid contains activation {0.2,0.35,0.5}, deactivation activation−0.15, minimum speech {0,250} ms, minimum silence {100,250} ms, padding {30,150} ms (24 settings). Frame probabilities are computed once per independent horizon, then replayed through the production segmenter. The best three pooled development union-activity miss+false-alarm settings were shortlisted; full S128 clustering evaluated those, the original baseline, and a shorter-window diagnostic. The final choice minimizes pooled speaker-time DER, then speaker-macro JER, then ordinal profile ID. This is a bounded shortlist search, not a claim of global optimality.

Oracle activity uses the union of the development RTTM intervals, without reference speaker labels or counts, as the clustering input. It is diagnostic only and never used on holdouts. Clustering windows, S128 settings, embedders and the absence of overlap detection are held fixed. Short-segment diagnostics lower the internal embedding cutoff from 100 to 45 ms; the production cutoff stays at 100 ms.

| Development meeting | Original DER / JER | Selected activity DER / JER | Oracle activity DER / JER |
|---|---:|---:|---:|
| ES2011a |37.7930% / 53.8078%|33.1123% / 48.2213%|18.2216% / 41.1434%|
| ES2011b |30.4951% / 31.5493%|23.5827% / 25.3175%|15.4738% / 17.9560%|

Neither baseline development meeting lost an entire region to the 100-ms embedding cutoff: zero regions, zero seconds. Ownership spans the whole retained region even when a trailing embedding window is too short. With oracle activity only one 50-ms region (ES2011a) was omitted. Lowering the cutoff recovered that 50 ms but changed the sampled embedding sequence and increased confusion: oracle DER became 21.1118% / 15.5025%, and baseline DER 40.1308% / 30.8970%. Short-window dropping is not the main missed-speech cause on these development meetings, and the shorter cutoff was not adopted.

Reproduce the bounded development search with `onespeech-eval diarize activity grid`, `shortlist`, `cluster`, and `freeze`; the command rejects nondevelopment IDs for fitting. `activity holdout` requires the frozen setting file and writes its SHA-256 into every row. No evaluation speaker labels enter selection.

Pooled development DER/JER changes from **33.3520% / 42.6786%** to **27.3133% / 36.7694%**. Baseline miss/false-alarm/confusion totals are 660.880 / 20.873 / 117.625 seconds; selected totals are 427.738 / 96.369 / 130.535 seconds. Oracle-activity totals are 306.990 / approximately 0 / 89.667 seconds (16.5495% DER). Of oracle misses, **306.940 seconds** are the unavoidable extra speaker-time during reference overlap for an exclusive single-speaker output; only 0.050 seconds come from the short-region cutoff. The original avoidable activity miss is about 353.940 seconds, larger than oracle-activity confusion. Oracle activity changes embedding segmentation as well as coverage, so this is a diagnostic breakdown rather than an additive guarantee for a replacement VAD.

The selected setting `T20-M250-S250-P150-C100` was frozen before held-out scoring (selection SHA-256 `c4d50356d42f1c4b0bd82a953ab94c9a1cae60f9b2e841e5a11b3b01bb49fed0`). Each of the four held-out meetings was then run once, without an oracle activity pass or further tuning. All speech-activity, model, clustering, precision and scoring details remain as above except the selected activity profile. Values below supersede the earlier activity floors for the helpers:

| Meeting | Selected DER / JER | Selected miss / FA / confusion (s) | Published bundle DER / JER | Bundle miss / FA / confusion (s) |
|---|---:|---:|---:|---:|
| ES2004a |28.8213% / 31.0004%|180.754 / 42.453 / 42.938|26.5359% / 29.6337%|176.510 / 49.150 / 19.380|
| ES2004b |21.1057% / 19.9543%|298.314 / 90.640 / 82.346|18.1017% / 19.2563%|282.540 / 84.210 / 37.470|
| R8001_M8004_MS801 |42.4480% / 42.3801%|771.504 / 19.706 / 78.652|34.5196% / 34.0168%|616.960 / 46.150 / 44.280|
| R8003_M8001_MS801 |27.2741% / 29.8615%|418.452 / 117.852 / 81.658|26.8887% / 28.5758%|414.640 / 127.700 / 66.890|

All four DER/JER pairs improve versus their earlier same-embedder floors. R8003 DER is within 0.5 points of the bundle; its JER and both metrics on the other meetings remain outside that bound. Single-file intervals remain unavailable. On AMI, the remaining bundle gap is primarily confusion (excess 23.558 / 44.876 seconds), not activity misses alone. R8001 still has a substantial missed-speech gap. No new segmentation/speaker-change model or overlap detector is added, and these different-system comparisons do not establish numerical parity.

## Cosine-centroid resegmentation

`CosineResegmenter` is an optional embedding-only algorithm, not a learned
speaker-change model or VBx replacement. It uses the existing 1.5-second
windows at 0.25-second hop; it does not extract finer audio windows or claim
sub-hop precision. VAD coverage and the midpoint ownership rule stay unchanged.
No extra waveform pass, model, biometric persistence or overlap detection is
introduced.

For each initial cluster, average the unit embeddings of stable windows, then
normalize the sum. A window is stable when its initial label and the labels of
`StableRadius` neighbors on either side agree, with no activity gap in that
support. If a cluster has no stable windows, use all its windows. A near-zero
centroid norm (less than 1e-12) leaves the whole initial labeling unchanged.
Cosine distance to these fixed centroids is the per-window assignment cost.
An exact minimum-cost path minimizes the sum of assignment costs plus
`SwitchPenalty` for each label change. Costs are per window, not per second;
the penalty is soft smoothing, not a hard minimum-turn duration. The path is
reset at a positive gap between embedding windows. Ties prefer staying, then
the lowest original label. Speaker IDs are canonicalized by the enclosing
diarizer after refinement. No speaker outside the initial labels is created;
a speaker can disappear, and the enclosing diarizer still enforces explicit
count/min/max constraints. It refuses a violating result rather than ignoring
a count request.

This is our own deterministic centroid/path design. It does not implement the
publisher's acoustic transformer locator: its public API returned rounded
change-point text, while its published card and paper do not specify a complete
inference graph. That learned locator is not loaded by these helpers.

Public algorithm defaults are penalty 0.1 and radius 2; these are starting
values, not a universal calibration. Limits: up to 1,000,000 windows, 32 distinct
speakers and dimensions 1–4096, matching the clustering bounds. Path storage
uses NK bytes for byte predecessors plus 4N bytes for output labels; centroid
accumulators require at most 16KD bytes, plus small DP buffers and collection
overhead. There is no additional PCM storage. Existing embedding-sequence
storage remains O(ND), explicitly capped by `MaxWindows`, not constant total
memory for arbitrarily long input. At the default N cap of 100,000, additional
path payload is at most 3.6 MB. Cancellation is checked per window and during
traceback.

### Selected file-helper profile

The file helpers use `CosineResegmenter(switchPenalty: 0.4, stableRadius: 2)`.
Low-level `ClusteringDiarizer` remains opt-in via its existing resegmenter
argument. The development-only search compared no resegmentation with all
15 combinations of radius {0,2,4} and penalty {0,0.05,0.1,0.2,0.4}, using the
already selected activity profile and S128 clustering unchanged. All windows
were the original 1.5 s / 0.25 s windows. Selection minimizes pooled development
speaker-time DER, then macro JER and ordinal setting ID; no resegmentation wins
a tie against the baseline. No AliMeeting Eval or AMI test labels were used.

| AMI development | No resegmentation DER / JER | Selected DER / JER |
|---|---:|---:|
| ES2011a |33.1123% / 48.2213%|31.5613% / 47.1127%|
| ES2011b |23.5827% / 25.3175%|22.1844% / 23.0626%|
| Pooled |27.3133% / 36.7694%|25.8552% / 35.0877%|

Pooled development confusion falls from 130.535 to 95.589 seconds. Miss
427.738 seconds and false alarm 96.369 seconds are unchanged: this algorithm
cannot recover activity or represent overlap. Profile `R2-P400` was frozen
before holdout inference, SHA-256
`1d57269880fdccc38b635a53cf19776041c178a06e3419c5bcfd42bda7d53f13`.
`onespeech-eval diarize activity refine-dev`, `refine-freeze` and
`refine-holdout` reproduce the search and fixed holdouts; the holdout command
checks the freeze digest and refuses to overwrite scores. Development embeddings
are cached only in RAM and cleared after each meeting.

The four complete holdouts were each run once with that frozen profile.
Precision, machine, model pins, activity settings and no-collar/overlap-inclusive
scoring are the same as the preceding activity rows. CPU float32, eight graph
threads; OneModel 1.0.0, OneAudio 1.7.2, SDK 10.0.403/runtime 10.0.14. The
reference remains the pinned ModelScope 1.40.2 published bundle, not this
centroid algorithm. No new reference inference was needed.

| Meeting | Before DER / JER | Refined DER / JER | Refined miss / FA / confusion (s) | Published bundle DER / JER |
|---|---:|---:|---:|---:|
| ES2004a |28.8213% / 31.0004%|28.2116% / 29.6213%|180.754 / 42.453 / 37.308|26.5359% / 29.6337%|
| ES2004b |21.1057% / 19.9543%|20.9156% / 19.7743%|298.314 / 90.640 / 78.102|18.1017% / 19.2563%|
| R8001_M8004_MS801 |42.4480% / 42.3801%|41.5741% / 41.1256%|771.504 / 19.706 / 60.743|34.5196% / 34.0168%|
| R8003_M8001_MS801 |27.2741% / 29.8615%|26.7902% / 28.9546%|418.452 / 117.852 / 70.694|26.8887% / 28.5758%|

All four improve over the preceding same-activity results, but the confusion
gap is not closed. Using a 0.5-point single-file difference bound, only ES2004a
JER and both R8003 metrics agree with the reference; ES2004b JER differs by
0.518 points and also misses. Single-file confidence intervals are unavailable.
These fixed meetings do not establish population quality. This is not parity
with the learned locator or the reference's full system.
