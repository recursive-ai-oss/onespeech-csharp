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

`ClusteringDiarizer` is the tier-1 voice-clustering implementation, using caller-supplied VAD and speaker-embedding roles. It **does not detect overlapped speech**: each speech interval gets only one speaker at a time. The overlap-aware upgrades, pyannote segmentation and Nemotron 3 Diarization, are coming as explicit tier-2 options; neither is loaded by these helpers. Channel diarization is a distinct exception because separated channels supply activity independently. There is no automatic embedder selection or quality-validated default checkpoint in this component yet.

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

An optional `EmbeddingResegmenter` receives borrowed windows/initial labels synchronously and returns a label per window before turn construction. The output must use only initial labels and respect count hints. No implementation is installed by default; in particular **VBx is deferred**. Landini §§2.1–2.3 requires a compatible pretrained PLDA mean, whitening/LDA transform and between-speaker variances. None is fitted or substituted here. Embeddings are never written to disk or retained by the diarizer after the call.

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

A black-box ModelScope 1.40.2 `speaker-diarization` pipeline probe rejected the pinned English CAM++ checkpoint at registry initialization (`speaker-verification` is not registered in the `speaker-diarization` group). Network was disabled to prevent unchecked auxiliary downloads. The documented bundle instead uses other VAD/Chinese/change-point checkpoints; no same-checkpoint reference was scored, and repository recipes were not opened or executed. Floors therefore stand without a comparative verdict.

The file pipeline also produced a 300-second AMI test excerpt with a deliberately synthetic **word-timing fixture**, not ASR: 211 timed clock probes, 116 assigned to a diarizer speaker and 95 explicitly unassigned. This proves clock/speaker document integration only, not recognized text, WER, alignment quality or correct speaker attribution. The synthetic separated-party stereo file test independently exercises two channel labels. See the evaluation commands and data manifests for reproducibility; holdout references were never used to choose parameters.

See [evaluation](eval.md) for no-collar, overlap-inclusive DER/JER and RTTM input.
