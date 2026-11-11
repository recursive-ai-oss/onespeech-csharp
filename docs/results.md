# Measured results and limits

The release revalidation table distinguishes full-split measurements from the earlier fixed-slice evidence below. The tables disclose incomplete datasets, qualified references and remaining gaps; these results are not a cross-hardware guarantee. Lower is better for all errors below. Scores are percentages unless otherwise stated. Unless noted, recognition quality rows use float32. Models and publisher revisions are pinned in the [catalog](models.md); see [notices](../THIRD-PARTY-NOTICES.md) for attribution. No speed comparison with a native implementation is made here.

**CI convention:** the release table labels absolute-score and paired-difference 95% intervals separately. The older slice tables give only intervals for **our score minus the reference score**, in percentage points. Recognition uses 1,000 utterance resamples, seed 1729, and the same `en/1` scoring normalizer on both sides. “Within noise” means the difference interval includes zero. With no computable CI, the comparison uses a 0.5-point bound. Slices must not be interpreted as full-test results.

## Release quality revalidation (2026-11-08–11)

The inference measurements started from core source `dd63517` through the public recognizer, live-session and embedding APIs. Later reader hardening changes invalid-input handling, not valid-model inference. Moonshine uses its default greedy decoding, 6.5 generated tokens per audio second, no biasing, and two CPU threads. The reference uses the pinned publisher weights, Transformers 5.22.2, float32, eager attention and deterministic generation. Speaker extraction uses eight threads on each side. Whisper int8 rows use temperature-zero greedy decoding without word timestamps on both sides; the reference uses beam/best-of one, no VAD filter and previous-text conditioning. Native CPU reservations changed from eight to two threads during the resumable run; precision and decoding controls did not change. No previously saved hypotheses or embeddings were substituted for the new inference runs.

| Model / task | Dataset | Metric | Ours [absolute 95% CI] | Reference / version | Paired difference CI | Verdict |
|---|---|---|---|---|---|---|
| Whisper large-v3-turbo fp32 | Full LibriSpeech test-clean, 2,620 utterances | WER % | 2.267 [2.093, 2.432] | 2.289 / openai-whisper 20250625 | [-0.136, +0.072] pp | Within noise |
| Whisper large-v3-turbo fp32 | Full LibriSpeech test-other, 2,939 utterances | WER % | 4.275 [4.032, 4.538] | 4.331 / openai-whisper 20250625 | [-0.139, +0.019] pp | Within noise |
| Whisper small fp32 | Full LibriSpeech test-clean, 2,620 utterances | WER % | 3.650 [3.452, 3.877] | 3.669 / openai-whisper 20250625 | [-0.083, +0.046] pp | Within noise |
| Whisper small fp32 | Full LibriSpeech test-other, 2,939 utterances | WER % | 7.852 [7.486, 8.236] | 7.924 / openai-whisper 20250625 | [-0.171, +0.031] pp | Within noise |
| Whisper large-v3-turbo mixed-int8 | Full LibriSpeech test-clean, 2,620 utterances | WER % | 2.291 [2.118, 2.461] | 2.355 / faster-whisper 1.2.1, CT2 4.8.5 int8 | [-0.367, +0.118] pp | Within noise |
| Whisper large-v3-turbo mixed-int8 | Full LibriSpeech test-other, 2,939 utterances | WER % | 4.479 [4.227, 4.734] | 4.503 / faster-whisper 1.2.1, CT2 4.8.5 int8 | [-0.514, +0.264] pp | Within noise |
| Whisper small mixed-int8 | Full LibriSpeech test-clean, 2,620 utterances | WER % | 3.731 [3.517, 3.965] | 3.927 / faster-whisper 1.2.1, CT2 4.8.5 int8 | [-0.645, +0.074] pp | Within noise |
| Whisper small mixed-int8 | Full LibriSpeech test-other, 2,939 utterances | WER % | 8.038 [7.658, 8.449] | 8.007 / faster-whisper 1.2.1, CT2 4.8.5 int8 | [-0.145, +0.185] pp | Within noise |
| Moonshine streaming-tiny, offline | Full LibriSpeech test-clean, 2,620 utterances | WER % | 4.811 [4.505, 5.138] | 4.811 / Transformers 5.22.2 | [0, 0] pp | Within noise |
| Moonshine streaming-tiny, offline | Full LibriSpeech test-other, 2,939 utterances | WER % | 12.616 [12.089, 13.133] | 12.620 / Transformers 5.22.2 | [-0.012, 0] pp | Within noise |
| Moonshine streaming-tiny mixed-int8, offline | Full LibriSpeech test-clean, 2,620 utterances | WER % | 4.771 [4.474, 5.090] | 4.811 / our fp32 | [-0.097, +0.011] pp | Within 0.5 pp of fp32 |
| Moonshine streaming-tiny mixed-int8, offline | Full LibriSpeech test-other, 2,939 utterances | WER % | 12.641 [12.107, 13.176] | 12.616 / our fp32 | [-0.103, +0.152] pp | Within 0.5 pp of fp32 |
| Moonshine streaming-tiny, live, 100 ms pushes | LibriSpeech test-clean, first 200 | WER % | 5.070 [4.016, 6.222] | No equivalent streaming-policy oracle | — | Floor |
| Moonshine streaming-tiny, live, 100 ms pushes | LibriSpeech test-other, first 200 | WER % | 18.185 [16.110, 20.255] | No equivalent streaming-policy oracle | — | Floor |
| Moonshine streaming-tiny, live, 100 ms pushes | LibriSpeech test-clean, first 200 | Mean finalisation latency, ms | 2254.6 [2000.6, 2559.8] | Whisper tiny DTW word-end estimates, openai-whisper 20250625 | — | Floor, not independent truth |
| Moonshine streaming-tiny, live, 100 ms pushes | LibriSpeech test-other, first 200 | Mean finalisation latency, ms | 1535.6 [1416.3, 1649.1] | Whisper tiny DTW word-end estimates, openai-whisper 20250625 | — | Floor, not independent truth |
| CAM++ English | LibriSpeech fixed 20,000 trials | EER % | 0.750 [not computed] | 0.750 / ModelScope 1.40.2 | [0, 0] pp | Within noise |
| CAM++ English | LibriSpeech fixed 20,000 trials | minDCF | 0.0379 [not computed] | 0.0379 / ModelScope 1.40.2 | [0, 0] | Within noise |
| ECAPA-TDNN | LibriSpeech fixed 20,000 trials | EER % | 1.100 [not computed] | 1.100 / SpeechBrain 1.1.2 | [0, 0] pp | Within noise |
| ECAPA-TDNN | LibriSpeech fixed 20,000 trials | minDCF | 0.0548 [not computed] | 0.0548 / SpeechBrain 1.1.2 | [0, 0] | Within noise |
| Whisper large-v3-turbo, language ID | FLEURS, five each English/French/German | Accuracy | 15/15 [not computed] | 15/15 / openai-whisper 20250625 | — | Information only |

Full-split Moonshine token/text identity is 5,558/5,559 sequences. The differing test-other recording is `6070-86744-0018` (33.645 seconds); it contains punctuation and word differences, not merely a different special-token representation. The within-noise WER result does **not** imply token-identical decoding on every recording.

Live finalisation latency means audio-time `Final` availability minus an edit-matched word's estimated end, not CPU inference latency. There are 4,427 matched tokens out of 4,685 reference tokens on clean, and 2,524 out of 3,086 on other. The calculation retains 35/32 negative estimates rather than clamping them. Latency intervals resample recordings 1,000 times with seed 1729. The 200-recording floors and the earlier 20-recording floors below have different populations and must not be compared as a regression test. Scoring the retained first-20 subset separately reproduces the earlier rounded floors: WER 1.70/8.33% and mean latency 3057/1617 ms.

Speaker trials contain 10,000 same-speaker and 10,000 different-speaker pairs, 73 speakers and 1,460 distinct trial utterances. Extraction covered 1,490 utterances, including additional parity probes. Paired bootstrap resamples trials within same/different strata, conditional on this fixed population; minDCF uses prior 0.01, unit costs and normalization. These rows do not replace the unavailable VoxCeleb1-O evaluation. Model revisions are Moonshine `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`, CAM++ English `032b8131a7ad812f87061955ca974c99060c5a03`, and ECAPA `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.

Environment: CPU-only x64, AMD Ryzen 9 7950X3D (16 physical / 32 logical CPUs; FMA, AVX2 and AVX-512 available), .NET SDK 10.0.403 / runtime 10.0.14, OneModel 1.2.0 and OneAudio 1.7.2. Language ID uses eight threads. Runs used a shared CPU queue; these are correctness measurements, not publishable speed rows.

**Coverage:** full Whisper turbo/small fp32 and int8, and full Moonshine fp32/int8 rows are complete. Each LibriSpeech comparison contains all 2,620 clean or 2,939 other utterances. Parakeet full-set, Qwen first-500 CPU-limited slices, five-recording long-form comparisons and timing revalidation are in the additional ASR section below. Full-set diarization and Nemotron streaming are reported in their separate section below. The fixed shortest-duration Earnings-22 selection is `4474955`, `4483046`, `4468919`, `4479524`, `4481221` (875.020, 1193.180, 1214.487, 1226.213, 1236.904 seconds). Its measured long-form rows and both temperature policies are reported below. Earlier misses below remain visible and are not superseded by unmeasured rows.

### Reader mutation campaigns

Each listed target completed a 600-second mutation budget (libFuzzer reported 601 seconds), with one fuzzer running at a time and four CPU cores reserved for publishing and mutation. SharpFuzz 2.2.0 instruments a disposable copy of the assembly containing the reader; libfuzzer-dotnet is release v2025.05.02.0904. Mutation input limits are 64 KiB, except SentencePiece (128 KiB) and Parakeet vocabulary (256 KiB). These bounded campaigns are not an exhaustive proof for all input sizes or dependency readers.

| Target | Mutation budget, s | Executions | Coverage features | Crashes |
|---|---|---|---|---|
| Transcript JSON | 600 | 28,843,861 | 1,596 | 0 |
| Model catalog | 600 | 24,003,517 | 865 | 0 |
| WAV / G.711 entry | 600 | 21,043,276 | 398 | 0 |
| General audio decode entry | 600 | 17,547,248 | 282 | 0 |
| Checkpoint ZIP | 600 | 27,408,863 | 1,045 | 0 |
| Checkpoint pickle index | 600 | 16,476,079 | 2,166 | 0 |
| Safetensors | 600 | 32,004,710 | 507 | 0 |
| SentencePiece | 600 | 27,466,894 | 1,081 | 0 |
| Moonshine tokenizer / configuration | 600 | 19,052,083 | 580 | 0 |
| Parakeet vocabulary / normalization | 600 | 2,599,800 | 251 | 0 |
| RTTM (evaluation tool) | 600 | 27,514,840 | 236 | 0 |
| Whisper byte-BPE files | 600 | 2,280,339 | 221 | 0 |
| Qwen byte-BPE files | 600 | 1,455,773 | 265 | 0 |
| Whisper configuration | 600 | 2,060,381 | 69 | 0 |
| Qwen configuration | 600 | 3,929,405 | 85 | 0 |
| Converter provenance (Whisper/Moonshine/Qwen) | 600 | 1,485,318 | 1,073 | 0 |
| Nemotron ASR processor configuration | 600 | 2,166,332 | 174 | 0 |
| Whisper generation configuration | 600 | 1,946,348 | 157 | 0 |

The checkpoint campaigns instrument the core assembly where the reader now resides, not the converter assembly. General audio decoding is exercised without the WAV-only prefilter; OneAudio itself is not instrumented. RTTM is an evaluation-tool reader, not a shipped core reader. No mutated-input crash was found. New hostile-input tests also cover malformed byte-BPE/configuration shapes, empty special tokens and invalid generation IDs/heads. The complete [reader inventory](model-data-readers.md) distinguishes owned readers from dependency parsers. One new harness startup failed because it touched an instrumented static initializer before fuzzer initialization; setup was corrected before the clean campaigns, with no input crash to minimize.

## Offline recognition

LibriSpeech rows are the first 200 sorted utterance IDs of each named split. Each row is separate; results are not averaged across splits or models.

| Model | Dataset slice | Metric / ours | Reference score | Black-box reference | Difference CI | Verdict |
|---|---|---|---|---|---|---|
| Whisper small | test-clean 200 | WER 2.503 | 2.674 | openai-whisper 20250625 | [-0.493, +0.066] | Within noise |
| Whisper small | test-other 200 | WER 10.280 | 10.312 | openai-whisper 20250625 | [-0.556, +0.554] | Within noise |
| Whisper large-v3-turbo | test-clean 200 | WER 1.476 | 1.561 | openai-whisper 20250625 | [-0.321, +0.129] | Within noise |
| Qwen3-ASR 0.6B | test-clean 200 | WER 1.754 | 1.711 | qwen-asr 0.0.6 | [0, +0.155] | Within noise |
| Qwen3-ASR 0.6B | test-other 200 | WER 5.628 | 5.628 | qwen-asr 0.0.6 | [0, 0] | Within noise |
| Qwen3-ASR 1.7B | test-clean 200 | WER 1.433 | 1.369 | qwen-asr 0.0.6 | [0, +0.233] | Within noise |
| Qwen3-ASR 1.7B | test-other 200 | WER 4.131 | 4.131 | qwen-asr 0.0.6 | [0, 0] | Within noise |
| Moonshine tiny | test-clean 200 | WER 3.72 | 3.72 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine tiny | test-other 200 | WER 14.54 | 14.54 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine base | test-clean 200 | WER 2.74 | 2.74 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine base | test-other 200 | WER 11.39 | 11.39 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine streaming-tiny | test-clean 200 | WER 3.64 | 3.64 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine streaming-tiny | test-other 200 | WER 14.96 | 14.96 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine streaming-small | test-clean 200 | WER 2.10 | 2.10 | Transformers 5.22.2 | [0, 0] | Within noise |
| Moonshine streaming-small | test-other 200 | WER 9.43 | 9.43 | Transformers 5.22.2 | [0, 0] | Within noise |
| Parakeet-TDT-0.6B-v3 | test-clean 200 | WER 2.053 | 2.053 | NeMo 3.0.0, greedy TDT | [0, 0] | Within noise |
| Parakeet-TDT-0.6B-v3 | test-other 200 | WER 4.424 | 4.424 | NeMo 3.0.0, greedy TDT | [0, 0] | Within noise |
| omniASR-CTC-300M | test-clean 200 | CER 1.2879 | 1.2879 | omnilingual-asr 0.2.0 | [0, 0] | Within noise |

The omniASR row tests the aligner's acoustic decoding, not general-purpose ASR coverage. Moonshine produced 1,600/1,600 identical token sequences across its eight rows; Parakeet produced identical tokens and frames across the two slices (14,365 tokens).

## Timing

Boundary errors are mean absolute start/end error in milliseconds. Confidence intervals were not supplied for these rows. A timing floor is a first measurement against independent truth, **not** a claim of equivalence to that truth.

| Method | Slice / paired words | Reference | Start / end ms | Within 50 ms start / end | Within 100 ms start / end | Verdict |
|---|---|---|---|---|---|---|
| Whisper-small DTW | test-clean 200 / 4,625 | openai-whisper 20250625 timestamps | 11.9 / 8.6 | 97.0 / 98.1% | 98.7 / 99.8% | Measured implementation difference |
| Qwen3-ForcedAligner | test-clean 200 | qwen-asr 0.0.6 toolkit | 28.7 / 18.9 | 90.8 / 98.1% | 94.4 / 98.1% | Post-processing parity gap; 67/4,634 untimed |
| Parakeet native | test-clean / 4,633 | NeMo 3.0.0 | 0 / 0 | 100 / 100% | 100 / 100% | Identical boundaries |
| omniASR CTC | test-clean 200 / 4,531 common words | MFA 3.4.3 | 52.35 / 50.33 | Not supplied | Not supplied | Floor |
| omniASR CTC | test-other 100 / 1,465 common words | MFA 3.4.3 | 49.77 / 53.00 | Not supplied | Not supplied | Floor |
| Our Whisper-small DTW | test-clean 200 / same common words | MFA 3.4.3 | 94.16 / 79.23 | Not supplied | Not supplied | Floor |
| Our Whisper-small DTW | test-other 100 / same common words | MFA 3.4.3 | 104.27 / 75.52 | Not supplied | Not supplied | Floor |
| Reference Whisper DTW | test-clean 200 / same common words | MFA 3.4.3 | 92.87 / 77.98 | Not supplied | Not supplied | Information only |
| Reference Whisper DTW | test-other 100 / same common words | MFA 3.4.3 | 103.96 / 75.80 | Not supplied | Not supplied | Information only |

MFA truth uses English MFA acoustic model/dictionary v3.0.0 with speaker adaptation disabled. The CTC aligner left 0/4,634 clean words and 9/1,562 other words untimed. Common-word comparisons exclude words not timed by every compared method; they do not erase these failures.

## Diarization

DER uses **no collar and scores overlap**. Values below are DER / JER percentages. Single-recording rows have no bootstrap CI. AMI uses Mix-Headset test recordings ES2004a/b; AliMeeting uses Eval far-field channel 1. The tier-1 method is Silero + CAM++ English (AMI) or Chinese (AliMeeting), spectral clustering and cosine resegmentation; its settings were selected on AMI development ES2011a/b, not these test recordings.

| Diarizer | Recording | Ours DER / JER | Reference DER / JER | Black-box reference | Verdict (DER / JER) |
|---|---|---|---|---|---|
| Tier-1 clustering | AMI ES2004a | 28.21 / 29.62 | 26.54 / 29.63 | ModelScope 1.40.2 CAM++ diarization bundle | Miss / within 0.5 |
| Tier-1 clustering | AMI ES2004b | 20.92 / 19.77 | 18.10 / 19.26 | ModelScope 1.40.2 CAM++ diarization bundle | Miss / miss |
| Tier-1 clustering | AliMeeting R8001_M8004_MS801 | 41.57 / 41.13 | 34.52 / 34.02 | ModelScope 1.40.2 CAM++ diarization bundle | Miss / miss |
| Tier-1 clustering | AliMeeting R8003_M8001_MS801 | 26.79 / 28.95 | 26.89 / 28.58 | ModelScope 1.40.2 CAM++ diarization bundle | Within 0.5 / within 0.5 |
| Nemotron 3 Diarization | AMI ES2004a | 26.90 / 29.08 | 26.90 / 29.09 | Transformers 5.22.2 | Within 0.5 / within 0.5 |
| Nemotron 3 Diarization | AMI ES2004b | 21.62 / 21.53 | 21.62 / 21.53 | Transformers 5.22.2 | Within 0.5 / within 0.5 |
| Nemotron 3 Diarization | AliMeeting R8001_M8004_MS801 | 18.64 / 18.03 | 18.64 / 18.03 | Transformers 5.22.2 | Within 0.5 / within 0.5 |
| Nemotron 3 Diarization | AliMeeting R8003_M8001_MS801 | 18.30 / 18.54 | 18.30 / 18.54 | Transformers 5.22.2 | Within 0.5 / within 0.5 |

Tier-1 overlap detection is absent. These misses are real limitations, not rounded-away successes. Short recordings (at most 60 seconds) now use cosine AHC: a separate seeded LibriSpeech short-recording check improved speaker-count accuracy from 9/24 to 23/24 (singles 12/12, pairs 11/12), with DER 5.85%. It is not an AMI/VoxConverse population estimate.

Online clustering floors with 100 ms pushes: ES2004a 47.08/54.66, ES2004b 26.88/27.71, R8001 56.39/53.94, R8003 39.41/42.64 DER/JER. Turn availability delay averaged 1.32–1.71 seconds. No equivalent reference-policy CI is available; live and whole-recording clustering are not interchangeable.

## Speaker embeddings

LibriSpeech test-clean + test-other, 20,000 balanced trials, 73 speakers, seed 1729. These are **identity-only trials on this fixed population**, not open-set VoxCeleb1-O results. The paired-trial CI is conditional on these speakers. minDCF uses prior 0.01, miss/false-alarm costs 1, normalized; scores use raw cosine.

| Model | Metric / ours | Reference | Reference version | Difference CI | Verdict |
|---|---|---|---|---|---|
| CAM++ English | EER 0.75%; minDCF 0.0379 | 0.75%; 0.0379 | ModelScope 1.40.2 | [0,0] both | Within noise |
| ECAPA-TDNN | EER 1.10%; minDCF 0.0548 | 1.10%; 0.0548 | SpeechBrain 1.1.2 | [0,0] both | Within noise |
| CAM++ Chinese | EER 2.87%; minDCF 0.1876 | 2.87%; 0.1876 | ModelScope 1.40.2 | [0,0] both | Information on English trials |

## Authorized local segmentation

The local segmentation-3.0 adapter uses the pinned publisher checkpoint with
float32 OneModel 1.2.0. On the first five ordinal LibriSpeech test-clean clips
(1089-134686-0000 through 0004), padded/truncated to 10 seconds, probability-space
max absolute error against pyannote.audio 4.0.7 / CPU Torch 2.11.0 is 9.19e-6;
the largest per-clip mean error is 2.64e-7. All 2,945 powerset frame decisions
match. This probe row uses four threads on x64 (AMD Ryzen 9 7950X3D);
it does not establish a bitwise or all-ISA guarantee. Raw log-probability errors and crafted low-variance cases are described in
[the segmentation contract](gated-segmentation.md); this is not bitwise parity.

The following are **floors for our segmentation-plus-CAM++ clustering policy**,
not community-1 equivalence. Both sides use fp32, no collar, overlap scored, and
the same scorer/truth. AMI v1.6.2 uses Mix-Headset audio; AliMeeting Eval uses
far-field channel 1. Each row is one recording, without a bootstrap CI.
Our side uses four CPU threads, independent 10 s chunks and the existing
spectral/AHC defaults; CAM++ English on AMI and Chinese on AliMeeting.
Community-1 is the publisher's complete pipeline (different embedder and PLDA/VBx),
run offline through pyannote.audio 4.0.7, for information only. Its ES2004a
reference uses four CPU threads; the remaining reference shards use eight to
fit the per-recording run bound, with unchanged model/pipeline parameters.

| Recording | Our DER / JER % | Community-1 DER / JER % | Status |
|---|---|---|---|
| AMI ES2004a | 20.62 / 25.99 | 18.76 / 22.74 | Floor; reference information |
| AMI ES2004b | 13.40 / 15.83 | 12.61 / 14.28 | Floor; reference information |
| AliMeeting R8001_M8004_MS801 | 26.46 / 30.41 | 22.02 / 24.55 | Floor; reference information |
| AliMeeting R8003_M8001_MS801 | 20.40 / 24.11 | 17.38 / 19.14 | Floor; reference information |

Community-1's native WeSpeaker/PLDA/VBx adapter is not implemented. Its published
PLDA does not require fitting, but its exact transform/dataflow contract has not
yet been established for an independent implementation. Its PLDA is never
silently substituted for a CAM++ backend. The tier-1 default is unchanged.


## Additional ASR, long-form and timing results

All WER rows: fp32, `en/1` normalizer, paired utterance (or file) bootstrap, 1000 resamples, seed 1729, scored by `onespeech-eval score`. "Delta" is ours minus reference; CIs are of the paired difference. Machine: AMD Ryzen 9 7950X3D (16C/32T, AVX2/FMA/AVX-512), shared worker at load 22-25, CPU only. .NET SDK 10.0.403, OneModel 1.2.0, OneAudio 1.7.2. Jobs ran as queue shards of 4-6 cores; no timing claim is made from them.

| Row | Data | Ours (95% CI) | Reference (pin) | Delta [CI] | Verdict |
|---|---|---|---|---|---|
| Parakeet-TDT-0.6B-v3 (tier 2), greedy TDT | LibriSpeech test-clean, all 2620 | 2.088% [1.931, 2.241] | 2.088% (NeMo 3.0.0, fp32) | 0.000 [0, 0] | within noise |
| same | test-other, all 2939 | 3.791% [3.575, 4.027] | 3.791% | 0.000 [0, 0] | within noise |
| Qwen3-ASR-1.7B | test-clean, first 500 sorted IDs | 1.326% [1.026, 1.677] | 1.260% (qwen-asr 0.0.6 on Transformers 4.57.6, fp32) | +0.066 [0, +0.175] | within noise |
| same | test-other, first 500 sorted IDs | 5.815% [5.077, 6.624] | 5.815% | 0.000 [0, 0] | within noise |
| Whisper large-v3-turbo long-form, temperature 0 on both sides | Earnings-22, 5 files (4474955, 4483046, 4468919, 4479524, 4481221) | 13.427% [9.701, 17.827] | 85.101% [49.0, 125.3] (openai-whisper 20250625 `transcribe()`) | -71.674 [-115.1, -33.6] | better than the reference pipeline; see below |
| same, default fallback temperatures, both sides (information) | same 5 files | 13.427% [9.701, 17.827] | 19.524% [17.243, 21.736] | -6.097 [-10.047, -2.409] | better than the reference pipeline |

Parakeet: all 99,653 (clean) and 97,717 (other) reference tokens are identical to NeMo's, and so are all emission frames (2620/2620 and 2939/2939 utterances). The raw text strings differ in 6 and 5 utterances only by spacing; the normalized scores are identical.

Qwen3-ASR-1.7B: the first 500 sorted IDs per split, not the full splits. The CPU reference costs about 115 utterances per hour on a shared host, so the full 5559 utterances did not fit. Raw text differs in 2 of 500 (clean) and 1 of 500 (other) utterances.

Long-form, per file (ours; reference at temperature 0; reference with default fallback), WER %:

| File | Ours | Reference T=0 | Reference default |
|---|---:|---:|---:|
| 4474955 | 8.473 | 58.386 | 21.960 |
| 4483046 | 21.119 | 58.837 | 20.711 |
| 4468919 | 13.790 | 37.137 | 22.137 |
| 4479524 | 10.070 | 140.140 | 15.944 |
| 4481221 | 10.869 | 136.665 | 16.805 |

Why the reference rows are large: at temperature 0, `transcribe()` with its default previous-text conditioning falls into repetition loops on these long recordings (reference output for 4479524 has 4000 words against 2122 reference words; for 4474955 one segment text repeats 346 times, for 4481221 1013 times). Our pipeline chunks at speech boundaries and does not condition across chunks, so it does not loop. Ours is identical with and without fallback on this set. These rows are better than the reference pipeline, not an equality or within-noise claim; the comparison is against the reference's own pipeline. The 4483046 default-fallback row is the one where the reference is better than ours-at-T=0 in per-file terms (20.711 vs 21.119, paired, no CI at n=1).

### Timing revalidation on current code

Mean absolute boundary error (ms) and share within 50/100 ms, pooled over timed words, against MFA 3.4.3 (English v3.0.0, speaker adaptation off) alignments of the same LibriSpeech slices (test-clean first 200, test-other first 100). Floors are the values recorded earlier; no row regressed.

| Method | Slice | Matched/timed | Start mean | End mean | Start within 50/100 % | End within 50/100 % | Recorded floor |
|---|---|---:|---:|---:|---|---|---|
| Timing order via `Speech.Transcribe` (turbo + Qwen3-ForcedAligner) | clean 200 | 4563/4563 | 26.17 | 26.95 | 91.41 / 98.79 | 90.88 / 98.53 | 26.2 / 27.0 |
| same | other 100 | 1499/1499 | 27.72 | 26.98 | 90.59 / 98.13 | 90.46 / 98.33 | none |
| omniASR-CTC via `Speech.Align` | clean 200 | 4634/4634 | 52.70 | 50.94 | 53.28 / 91.11 | 60.25 / 88.82 | 52.70 / 50.94 (identical) |
| same | other 100 | 1571/1562 | 49.69 | 53.37 | 57.11 / 92.25 | 59.73 / 86.81 | 49.69 / 53.37 (identical) |
| Whisper small DTW (ours) | clean 200 | 4540/4540 | 94.24 | 79.28 | 49.07 / 71.15 | 46.34 / 69.25 | identical |
| same | other 100 | 1470/1470 | 104.07 | 75.49 | 49.80 / 72.86 | 47.89 / 71.77 | identical |
| openai-whisper DTW (information, unchanged archived run) | clean / other | 4531 / 1469 | 92.87 / 103.89 | 77.98 / 75.80 | | | |

Qwen3-ForcedAligner vs the toolkit, clean 200 (4634 words): start/end mean 28.73/18.90 ms; start within 50/100 ms 90.80/94.37 %, end 98.07/98.10 %; 67 words untimed by ours, 0 by the toolkit. Identical to the recorded row (post-processing parity remains a documented gap).

Numbers are word-pooled and cover the matched words of each method's own transcript, so the turbo row (its own recognized words) and the CTC row (reference text) have different word sets, as in the earlier floors.

## Diarization and cache-aware streaming — full-set evaluation

Float32 CPU evaluation on an AMD Ryzen 9 7950X3D (x64; AVX2/FMA/AVX512 available), .NET SDK 10.0.403/runtime 10.0.14, OneModel 1.2.0 and OneAudio 1.7.2. These are quality and audio-clock availability measurements, not throughput benchmarks.

### Data and scoring

- **AMI:** all 16 Full-corpus-ASR test meetings: EN2002a–d, ES2004a–d, IS1009a–d and TS3003a–d. Original Mix-Headset audio, 16 kHz mono, word annotations v1.6.2; positive silence gaps retained. CC-BY-4.0.
- **AliMeeting:** all eight Eval far-field recordings, first input channel, 16 kHz mono, nonempty TextGrid intervals. CC-BY-SA-4.0, evaluation only.
- DER has **no collar and scores overlap**. Dataset DER pools reference speaker-seconds; JER averages reference-speaker errors. Paired 95% intervals use 1,000 meeting resamples, seed 1729. Meetings within an AMI series share participants; these are meeting, not independent-speaker, resamples.
- **default:** the unchanged file-helper policy, Silero VAD + CAM++ English for AMI/Chinese for AliMeeting + spectral/AHC clustering and cosine resegmentation. It does **not detect overlap**. Reference: released ModelScope 1.40.2 CAM++ bundle, pinned local auxiliaries, float32 CPU, four threads. No test-set fitting.
- **nemotron:** Nemotron 3 Diarization, explicitly selected tier 2, compared with Transformers 5.22.2 at the same publisher pin and precision; four threads on both sides.
- **pyannote:** the segmentation-3.0/CAM++ slot, four threads, a **floor**, not a community-1 reimplementation. The pyannote.audio 4.0.7 community-1 pipeline is information only: its embedding, stitching and PLDA/VBx backend differ. Authorized local snapshots, offline; eight CPU threads, or sixteen on five recordings that exceeded the bounded eight-thread reservation. Neither tier-2 path is a default.

### Raw and qualified reference results

The ModelScope public output for EN2002c contains one reversed interval, `[2043.28, 2043.10, speaker 1]`, among 513 returned turns. The raw output and strict error are preserved. A negative-duration interval has no valid RTTM representation; a numeric raw reference score is therefore unavailable. An explicit qualified projection omits **only this interval** and retains the other 512 turns. It never swaps endpoints or invents timing.

| EN2002c reference treatment | Our DER / JER % | Reference DER / JER % | Interpretation |
|---|---:|---:|---|
| Raw public output, strict serialization | 32.8114 / 32.7066 | Unscorable: reversed interval | Unqualified full-AMI reference score/CI unavailable |
| Explicitly qualified projection | 32.8114 / 32.7066 | 30.5655 / 32.7222 | Full-AMI comparison below is conditional on this one omission |

All other ModelScope recordings were valid. Dataset-level tier-1 differences include zero in their paired intervals, but that does not hide substantial individual-meeting differences: for example EN2002b is +6.63 DER/+15.59 JER points. Every meeting is listed below. Nemotron is within paired noise on both complete sets. The pyannote slot remains below community-1 quality: AMI differences are +3.68 DER/+8.12 JER points and AliMeeting +4.65/+5.67 points, with positive paired intervals; those are disclosed information-baseline gaps, not equivalence passes.

### Nemotron 3.5 ASR streaming 0.6B

First 200 ordinal LibriSpeech test-clean/test-other utterances, float32 greedy English decoding. Reference: Transformers 5.22.2 CPU, two threads; OneSpeech four threads, actual published live-event policy. Known utterance boundaries isolate recognition from endpoint-detector differences. Normalizer **en/1**, 1,000 paired utterance resamples, seed 1729.

All **800** native/reference utterance/chunk pairs have identical lexical token sequences. Both sides were freshly generated. Paired WER differences and their 95% intervals are **[0,0]** on every row.

| Chunk ms | Split / utterances | WER ours = reference % | Each WER 95% CI % | Token-identical pairs |
|---:|---|---:|---|---:|
|80|test-clean / 200|2.888|[2.264, 3.589]|200/200|
|80|test-other / 200|9.824|[8.331, 11.312]|200/200|
|560|test-clean / 200|2.802|[2.223, 3.493]|200/200|
|560|test-other / 200|8.295|[6.983, 9.672]|200/200|

**Stable-word availability** closes a word at the next word boundary or EOF. Both engines use the same saved offline 80-ms RNN-T acoustic endpoints as a timing proxy, not forced-alignment truth. Signed negative delays are retained. Latency means are over matched words; unmatched counts are shown. This is audio-time availability, not CPU speed or a learned endpointing-policy comparison. Publication remains approximately **55–58 ms later** in OneSpeech, so latency parity is not claimed.

| Chunk ms | Split | Ours mean / median ms | Reference mean / median ms | Mean gap ms | Matched / unmatched words |
|---:|---|---|---|---:|---|
|80|test-clean|258.541 / 240.000|200.100 / 179.000|+58.441|4576 / 64|
|80|test-other|217.315 / 160.000|160.448 / 99.000|+56.867|2871 / 143|
|560|test-clean|405.200 / 400.000|347.623 / 339.000|+57.577|4594 / 48|
|560|test-other|348.760 / 320.000|293.566 / 259.000|+55.194|2895 / 123|

Actual EOF Final availability relative to the last common offline word end is identical for both engines and chunk sizes: test-clean mean/median **33.875/0 ms**, test-other **12.925/0 ms**. Utterance EOF is known in this experiment. These descriptive latency statistics have no bootstrap interval.

### Diarized transcript sanity check

Whisper small through `Speech.Transcribe`, English, AMI ES2004a's first 300 seconds: en/1 WER **16.0584%** on 411 reference words. After optimal one-to-one word-count speaker mapping, **287/344 = 83.4302%** of all Word objects have the correct maximum-overlap reference speaker, or **287/321 = 89.4081%** among eligible timed/reference-covered words. There are **29 Unassigned**, zero untimed words and zero overlap flags. This speaker measure is temporal attribution, not lexical correctness.

Timing provenance is 333 ForcedAligner, five CtcAligner and six Dtw words. The earlier all-word sanity figure was 84.3%; the current value is 83.43%, while WER remains 16.06%. The timing chain has changed since the earlier DTW-only measurement; no improvement claim is made.

### Complete-set and per-meeting diarization rows

All rates are percentages. Difference intervals and single-meeting differences are percentage points. The single-meeting 0.5-point diagnostic is descriptive; complete-set verdicts use paired meeting intervals. Pyannote/community-1 rows are information, not native-equivalence bars.

### ami — complete-set comparisons

| System | DER ours / reference % | DER difference [95% CI] pp | JER ours / reference % | JER difference [95% CI] pp | Interpretation |
|---|---|---|---|---|---|
| default (N=16) | 28.605698 / 26.996987 | +1.608711 [-0.645170, +3.608149] | 32.731323 / 31.691698 | +1.039624 [-2.895311, +5.636265] | within paired noise; qualified reference |
| nemotron (N=16) | 25.956223 / 25.956130 | +0.000093 [-0.000632, +0.001013] | 28.229055 / 28.228935 | +0.000120 [-0.000768, +0.001258] | within paired noise |
| pyannote (N=16) | 20.727578 / 17.049233 | +3.678345 [+2.154713, +5.615498] | 30.307162 / 22.187481 | +8.119681 [+4.497820, +12.028704] | floor; community-1 information only |

### ami — every meeting

| System | Recording | DER ours / reference % | JER ours / reference % | Delta DER / JER pp | Single-meeting diagnostic |
|---|---|---|---|---|---|
| default | EN2002a | 38.1571 / 37.9171 | 38.7891 / 39.5974 | +0.2400 / -0.8083 | outside 0.5 pp: JER |
| default | EN2002b | 41.1616 / 34.5341 | 53.2373 / 37.6481 | +6.6275 / +15.5893 | outside 0.5 pp: DER, JER |
| default | EN2002c | 32.8114 / 30.5655 | 32.7066 / 32.7222 | +2.2460 / -0.0156 | outside 0.5 pp: DER; one malformed oracle turn omitted |
| default | EN2002d | 36.4656 / 43.8972 | 38.4239 / 54.0351 | -7.4316 / -15.6112 | outside 0.5 pp: DER, JER |
| default | ES2004a | 28.2116 / 26.5359 | 29.6213 / 29.6337 | +1.6757 / -0.0125 | outside 0.5 pp: DER |
| default | ES2004b | 20.9156 / 18.1098 | 19.7743 / 19.3180 | +2.8059 / +0.4564 | outside 0.5 pp: DER |
| default | ES2004c | 19.2390 / 19.2674 | 19.7516 / 20.3014 | -0.0283 / -0.5498 | outside 0.5 pp: JER |
| default | ES2004d | 29.8934 / 28.3500 | 30.8881 / 32.7533 | +1.5434 / -1.8652 | outside 0.5 pp: DER, JER |
| default | IS1009a | 28.8166 / 27.4062 | 38.1760 / 41.7104 | +1.4104 / -3.5344 | outside 0.5 pp: DER, JER |
| default | IS1009b | 16.3341 / 16.6866 | 17.3990 / 17.9862 | -0.3525 / -0.5871 | outside 0.5 pp: JER |
| default | IS1009c | 15.4235 / 14.5912 | 14.2788 / 15.2765 | +0.8323 / -0.9978 | outside 0.5 pp: DER, JER |
| default | IS1009d | 24.0613 / 24.9293 | 28.7483 / 31.4794 | -0.8680 / -2.7311 | outside 0.5 pp: DER, JER |
| default | TS3003a | 30.9360 / 24.5841 | 65.1388 / 65.9223 | +6.3519 / -0.7835 | outside 0.5 pp: DER, JER |
| default | TS3003b | 29.5800 / 19.3441 | 42.8439 / 17.2337 | +10.2358 / +25.6102 | outside 0.5 pp: DER, JER |
| default | TS3003c | 24.2966 / 20.0232 | 21.6058 / 18.7358 | +4.2734 / +2.8700 | outside 0.5 pp: DER, JER |
| default | TS3003d | 33.0900 / 31.1031 | 32.3121 / 32.9713 | +1.9869 / -0.6592 | outside 0.5 pp: DER, JER |
| nemotron | EN2002a | 29.8965 / 29.8961 | 31.4741 / 31.4739 | +0.0004 / +0.0003 | within 0.5 pp |
| nemotron | EN2002b | 30.1470 / 30.1465 | 30.7041 / 30.7037 | +0.0004 / +0.0004 | within 0.5 pp |
| nemotron | EN2002c | 29.1778 / 29.1778 | 29.3575 / 29.3575 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | EN2002d | 33.2286 / 33.2286 | 34.7778 / 34.7778 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | ES2004a | 26.8965 / 26.9008 | 29.0843 / 29.0874 | -0.0043 / -0.0031 | within 0.5 pp |
| nemotron | ES2004b | 21.6238 / 21.6238 | 21.5331 / 21.5331 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | ES2004c | 20.3478 / 20.3500 | 20.2420 / 20.2445 | -0.0022 / -0.0026 | within 0.5 pp |
| nemotron | ES2004d | 22.4500 / 22.4500 | 23.5946 / 23.5946 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | IS1009a | 27.6462 / 27.6462 | 46.7131 / 46.7131 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | IS1009b | 14.9801 / 14.9801 | 15.5914 / 15.5914 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | IS1009c | 15.5322 / 15.5322 | 15.5646 / 15.5646 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | IS1009d | 19.7187 / 19.7187 | 22.7378 / 22.7378 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | TS3003a | 34.9082 / 34.9082 | 44.7680 / 44.7680 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | TS3003b | 25.8803 / 25.8742 | 25.7980 / 25.7907 | +0.0060 / +0.0073 | within 0.5 pp |
| nemotron | TS3003c | 29.9417 / 29.9422 | 29.5242 / 29.5245 | -0.0005 / -0.0003 | within 0.5 pp |
| nemotron | TS3003d | 31.4813 / 31.4813 | 30.4825 / 30.4825 | +0.0000 / +0.0000 | within 0.5 pp |
| pyannote | EN2002a | 25.2994 / 21.9055 | 31.1313 / 26.5933 | +3.3940 / +4.5380 | information, not parity |
| pyannote | EN2002b | 25.1858 / 20.7423 | 32.2769 / 26.1589 | +4.4435 / +6.1180 | information, not parity |
| pyannote | EN2002c | 19.7641 / 17.5378 | 22.7784 / 19.9288 | +2.2263 / +2.8497 | information, not parity |
| pyannote | EN2002d | 34.8360 / 23.1721 | 50.5422 / 27.7051 | +11.6639 / +22.8371 | information, not parity |
| pyannote | ES2004a | 20.6223 / 18.7569 | 25.9899 / 22.7352 | +1.8654 / +3.2547 | information, not parity |
| pyannote | ES2004b | 13.3961 / 12.6053 | 15.8318 / 14.2777 | +0.7908 / +1.5541 | information, not parity |
| pyannote | ES2004c | 13.5521 / 12.3476 | 15.7594 / 13.6303 | +1.2045 / +2.1292 | information, not parity |
| pyannote | ES2004d | 20.7726 / 18.6732 | 27.6010 / 24.3477 | +2.0994 / +3.2533 | information, not parity |
| pyannote | IS1009a | 28.0381 / 19.6477 | 53.5455 / 33.2918 | +8.3903 / +20.2537 | information, not parity |
| pyannote | IS1009b | 15.4046 / 15.1910 | 19.2789 / 15.9595 | +0.2136 / +3.3194 | information, not parity |
| pyannote | IS1009c | 12.4533 / 10.0505 | 15.2045 / 11.7915 | +2.4028 / +3.4130 | information, not parity |
| pyannote | IS1009d | 25.5250 / 16.4544 | 47.5072 / 25.4566 | +9.0705 / +22.0506 | information, not parity |
| pyannote | TS3003a | 21.4821 / 17.1439 | 59.8290 / 39.2217 | +4.3382 / +20.6074 | information, not parity |
| pyannote | TS3003b | 13.6168 / 11.9684 | 16.3609 / 13.8352 | +1.6484 / +2.5257 | information, not parity |
| pyannote | TS3003c | 15.2336 / 13.3405 | 16.7263 / 13.5044 | +1.8931 / +3.2219 | information, not parity |
| pyannote | TS3003d | 25.8165 / 21.0753 | 32.6690 / 25.9974 | +4.7412 / +6.6716 | information, not parity |

### alimeeting — complete-set comparisons

| System | DER ours / reference % | DER difference [95% CI] pp | JER ours / reference % | JER difference [95% CI] pp | Interpretation |
|---|---|---|---|---|---|
| default (N=8) | 35.487475 / 37.264385 | -1.776910 [-7.743750, +2.860047] | 39.678194 / 42.152640 | -2.474446 [-13.124693, +4.145233] | within paired noise |
| nemotron (N=8) | 17.520012 / 17.522146 | -0.002135 [-0.005671, +0.000577] | 18.115182 / 18.116630 | -0.001449 [-0.004958, +0.000871] | within paired noise |
| pyannote (N=8) | 25.079101 / 20.431073 | +4.648028 [+1.130799, +8.780060] | 31.380495 / 25.710263 | +5.670231 [+0.666037, +11.195585] | floor; community-1 information only |

### alimeeting — every meeting

| System | Recording | DER ours / reference % | JER ours / reference % | Delta DER / JER pp | Single-meeting diagnostic |
|---|---|---|---|---|---|
| default | R8001_M8004_MS801 | 41.5741 / 34.5196 | 41.1256 / 34.0168 | +7.0545 / +7.1088 | outside 0.5 pp: DER, JER |
| default | R8003_M8001_MS801 | 26.7902 / 26.8887 | 28.9546 / 28.5758 | -0.0985 / +0.3788 | within 0.5 pp |
| default | R8007_M8010_MS803 | 48.2493 / 64.5159 | 54.9748 / 83.9990 | -16.2667 / -29.0242 | outside 0.5 pp: DER, JER |
| default | R8007_M8011_MS806 | 34.7856 / 33.3411 | 48.5413 / 42.2086 | +1.4445 / +6.3327 | outside 0.5 pp: DER, JER |
| default | R8008_M8013_MS807 | 62.1732 / 60.0237 | 61.9760 / 61.6746 | +2.1495 / +0.3013 | outside 0.5 pp: DER |
| default | R8009_M8018_MS809 | 16.9636 / 17.0693 | 17.4936 / 17.9012 | -0.1057 / -0.4077 | within 0.5 pp |
| default | R8009_M8019_MS810 | 20.5389 / 20.8256 | 23.1732 / 22.2419 | -0.2868 / +0.9313 | outside 0.5 pp: JER |
| default | R8009_M8020_MS810 | 15.6915 / 15.8842 | 15.1541 / 16.6524 | -0.1927 / -1.4983 | outside 0.5 pp: JER |
| nemotron | R8001_M8004_MS801 | 18.6372 / 18.6372 | 18.0251 / 18.0251 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | R8003_M8001_MS801 | 18.3003 / 18.3003 | 18.5405 / 18.5405 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | R8007_M8010_MS803 | 21.8159 / 21.8159 | 24.4054 / 24.4054 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | R8007_M8011_MS806 | 13.1157 / 13.1136 | 14.3962 / 14.3934 | +0.0021 / +0.0027 | within 0.5 pp |
| nemotron | R8008_M8013_MS807 | 18.2221 / 18.2221 | 18.6807 / 18.6807 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | R8009_M8018_MS809 | 16.7428 / 16.7428 | 16.7129 / 16.7129 | +0.0000 / +0.0000 | within 0.5 pp |
| nemotron | R8009_M8019_MS810 | 16.3699 / 16.3793 | 16.6533 / 16.6632 | -0.0094 / -0.0099 | within 0.5 pp |
| nemotron | R8009_M8020_MS810 | 14.3181 / 14.3313 | 14.3182 / 14.3319 | -0.0132 / -0.0136 | within 0.5 pp |
| pyannote | R8001_M8004_MS801 | 26.4572 / 22.0170 | 30.4064 / 24.5484 | +4.4402 / +5.8580 | information, not parity |
| pyannote | R8003_M8001_MS801 | 20.3952 / 17.3844 | 24.1139 / 19.1369 | +3.0107 / +4.9770 | information, not parity |
| pyannote | R8007_M8010_MS803 | 37.6432 / 34.2414 | 47.6091 / 48.9949 | +3.4018 / -1.3858 | information, not parity |
| pyannote | R8007_M8011_MS806 | 28.9305 / 20.1727 | 46.2390 / 28.1000 | +8.7578 / +18.1390 | information, not parity |
| pyannote | R8008_M8013_MS807 | 25.1735 / 28.4585 | 29.8080 / 32.9240 | -3.2850 / -3.1161 | information, not parity |
| pyannote | R8009_M8018_MS809 | 14.5736 / 11.9651 | 16.6198 / 13.0970 | +2.6085 / +3.5228 | information, not parity |
| pyannote | R8009_M8019_MS810 | 26.5512 / 8.7164 | 24.3318 / 9.2037 | +17.8348 / +15.1281 | information, not parity |
| pyannote | R8009_M8020_MS810 | 10.0264 / 7.8391 | 9.8560 / 8.1313 | +2.1874 / +1.7247 | information, not parity |

## Speed

Measured 2026-11-08 using the [structural harness](../bench/OneSpeech.Bench/README.md), measurement version 5, one repetition per row. These are our own CPU measurements, not comparisons with another implementation. All models use fp32 at the publisher revisions pinned in the catalog.

- **x64:** AMD Ryzen 9 7950X3D, 16 physical / 32 logical CPUs, AVX2/AVX512F/FMA, Linux Mint 22.3.
- **arm64:** Apple M2, 8 logical CPUs, 8 GiB RAM, AdvSimd, macOS 27.0.1.
- Both use .NET SDK 10.0.403, runtime 10.0.14, OneModel 1.2.0 and OneAudio 1.7.2; no GPU. The process CPU-count limit equals the row's thread count. Silero retains its production one-thread setting.

Each filtered row ran in an exclusive `bench` queue job bounded to 20 minutes. Models were loaded and warmed on a **different** ten-second clip before timing. Load time is separate; download, conversion and verification are excluded. End-to-end is encoded audio to Transcript using retained model instances and the production recognition, deferred alignment, clustering and merge algorithms. This is **not** cold `Speech.Transcribe` latency: that helper releases sessions between stages, while this warm-throughput harness retains them and can use more memory.

Inputs are deterministic LibriSpeech test-clean concatenations of 60, 600 and 3600 seconds, without inserted silence; clustering uses the 1049.355-second AMI ES2004a Mix-Headset recording. These constructed inputs make no quality claim. `align` is omniASR-CTC forced alignment; `diarize` is Silero + CAM++ English clustering; `live` is Moonshine streaming-tiny with 100 ms chunks; `long` is the full file pipeline with Whisper tiny on 3600 seconds.

The structural bar is end-to-end / summed graph wall time **≤1.15**. RTF is elapsed seconds / audio seconds: **≤1 means at least real time**. RSS is decimal GB, sampled every five seconds; “steady” discards the first 30 seconds but includes phase changes, so its range is not evidence of a constant-memory plateau. Short rows without steady samples fall back to the sampled peak. Live working B/chunk excludes graph-runtime allocations and immutable emitted payloads (500 measured chunks after 100 warm chunks); it is not total process allocation.

| Machine | Row | Threads | Load s | E2E s | Graph s | Ratio | Peak / steady GB | RTF | B/chunk | Bench job |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| x64 | turbo60 | 1 | 9.80 | 102.73 | 101.20 | 1.015 | 13.21 / 11.94–13.21 | 1.712 | — | 008051 |
| x64 | turbo60 | 4 | 8.89 | 37.17 | 36.13 | 1.029 | 13.23 / 12.63–13.23 | 0.619 | — | 008053 |
| x64 | turbo600 | 1 | 9.53 | 799.19 | 790.48 | 1.011 | 13.60 / 11.86–13.60 | 1.332 | — | 008054 |
| x64 | turbo600 | 4 | 9.13 | 313.43 | 305.18 | 1.027 | 13.54 / 11.90–13.54 | 0.522 | — | 008061 |
| x64 | small | 1 | 7.10 | 44.06 | 42.67 | 1.032 | 9.86 / 9.12–9.86 | 0.734 | — | 008029 |
| x64 | small | 4 | 6.35 | 21.63 | 20.77 | 1.041 | 9.78 / 9.78–9.78 | 0.361 | — | 008040 |
| x64 | align | 1 | 1.88 | 34.27 | 33.99 | 1.008 | 5.02 / 4.78–4.78 | 0.571 | — | 008042 |
| x64 | align | 4 | 1.81 | 9.99 | 9.72 | 1.027 | 5.04 / 5.04–5.04 | 0.166 | — | 008044 |
| x64 | diarize | 1 | 0.14 | 70.78 | 68.76 | 1.029 | 0.47 / 0.46–0.47 | 0.067 | — | 008045 |
| x64 | diarize | 4 | 0.14 | 46.77 | 44.76 | 1.045 | 0.47 / 0.47–0.47 | 0.045 | — | 008048 |
| x64 | live | 1 | 0.62 | 21.25 | 21.14 | 1.005 | 0.74 / 0.74–0.74 | 0.354 | 0 | 008050 |
| x64 | long | 8 | 5.43 | 627.14 | 579.96 | 1.081 | 9.95 / 7.94–9.95 | 0.174 | — | 008063 |
| arm64 | small | 1 | 33.94 | 84.13 | 82.43 | 1.021 | 2.77 / 1.32–2.77 | 1.402 | — | 002027 |
| arm64 | small | 4 | 31.39 | 38.48 | 37.38 | 1.029 | 2.43 / 1.86–1.86 | 0.641 | — | 002029 |
| arm64 | align | 1 | 1.00 | 54.76 | 54.58 | 1.003 | 3.70 / 2.78–3.29 | 0.913 | — | 002030 |
| arm64 | align | 4 | 0.94 | 17.85 | 17.58 | 1.015 | 3.27 / 3.27–3.27 | 0.298 | — | 002031 |
| arm64 | diarize | 1 | 0.10 | 91.42 | 88.66 | 1.031 | 0.46 / 0.44–0.46 | 0.087 | — | 002032 |
| arm64 | diarize | 4 | 0.09 | 54.63 | 51.75 | 1.056 | 0.46 / 0.45–0.46 | 0.052 | — | 002033 |
| arm64 | live | 1 | 0.39 | 29.29 | 29.17 | 1.004 | 0.76 / 0.75–0.76 | 0.488 | 0 | 002034 |
| arm64 | long | 4 | 29.68 | 1144.50 | 1087.01 | 1.053 | 4.79 / 0.48–4.79 | 0.318 | — | 002035 |

The load-average table below reports the one-minute component before / during / after each timed call. The during interval includes the workload itself. Queue startup CPU snapshots showed no other CPU-bound user workload; desktop and OS services remained active. Large initial load averages can reflect decaying earlier activity, not concurrent CPU utilization. Full 1/5/15-minute samples and queue logs were retained with the measurement records.

| Machine | Row / threads | Load before | During min–max | Load after |
|---|---|---:|---|---:|
| x64 | turbo60 / 1 | 1.62 | 1.23–1.68 | 1.25 |
| x64 | turbo60 / 4 | 1.70 | 1.70–3.10 | 3.10 |
| x64 | turbo600 / 1 | 1.95 | 1.12–1.95 | 1.27 |
| x64 | turbo600 / 4 | 1.59 | 1.54–3.92 | 3.37 |
| x64 | small / 1 | 17.04 | 9.25–17.04 | 9.25 |
| x64 | small / 4 | 6.97 | 5.65–6.97 | 5.65 |
| x64 | align / 1 | 4.27 | 3.26–4.27 | 3.26 |
| x64 | align / 4 | 2.86 | 2.79–2.95 | 2.79 |
| x64 | diarize / 1 | 2.57 | 1.69–2.57 | 1.80 |
| x64 | diarize / 4 | 1.75 | 1.75–2.87 | 2.59 |
| x64 | live / 1 | 2.38 | 1.91–2.38 | 1.91 |
| x64 | long / 8 | 2.73 | 2.73–7.20 | 4.94 |
| arm64 | small / 1 | 2.08 | 2.07–2.55 | 2.55 |
| arm64 | small / 4 | 2.51 | 2.28–3.30 | 3.30 |
| arm64 | align / 1 | 3.01 | 2.19–3.01 | 2.19 |
| arm64 | align / 4 | 2.42 | 2.42–3.08 | 3.08 |
| arm64 | diarize / 1 | 3.15 | 2.33–3.22 | 2.33 |
| arm64 | diarize / 4 | 2.33 | 2.33–3.27 | 3.27 |
| arm64 | live / 1 | 3.27 | 2.82–3.27 | 2.92 |
| arm64 | long / 4 | 2.53 | 2.49–6.38 | 4.67 |

**Results:** every measured structural ratio passes (maximum 1.081 on x64, 1.056 on arm64). The tier-1 live default meets real time on one core on both machines, with 0 B/chunk of our working allocation. This does not imply that all file models run in real time on one core: turbo's measured RTF is 1.712 on 60 seconds and 1.332 on 600 seconds.

**Working-set qualification:** the one-hour run is not literally flat on x64. After discarding the first minute of each phase, recognition's first/last fifth means are 8.001/8.013 GB and diarization's are 9.356/9.376 GB, but alignment rises from 8.837 to 9.772 GB (+10.6%, fitted slope +6.47 MB/s). The whole-run first/last fifth means are 8.003/9.365 GB. This row therefore does **not** establish a strict flat-RSS bar; it also cannot distinguish bounded buffer-pool residency from input-length growth. On arm64, the corresponding recognition means are 0.841/0.865 GB, alignment 4.352/2.026 GB, and diarization 0.610/0.525 GB. No tuning or forced collection was used to alter these rows. These observations should not be turned into a universal constant-memory guarantee.

**Missing row:** Qwen3-ASR-1.7B was not measured. The approximately 62 GiB shared model/data tree plus its upstream and converted files would exceed the run's approximately 70 GiB storage budget. No Qwen speed claim or cross-model default re-selection is inferred from this omission. The one-hour memory rows use tiny; turbo is covered only through 600 seconds. No int8 speed row was measured here.


## Model choices and remaining evidence

- **Files:** Whisper large-v3-turbo remains the default. Its test-other WER is 5.758% in the file-model comparison; Qwen3-ASR 1.7B is 4.131%, a turbo-minus-Qwen difference of +1.63 points (paired CI [1.02, 2.31]). Qwen 1.7B is available by name for accuracy. The fixed-slice CPU comparison favoured turbo's resource requirements; the completed structural measurements and their limits are reported in the Speed section.
- **Live:** Moonshine streaming-tiny is the default. On 20-file clean/other streaming slices, its WER floor is 1.70/8.33%, with mean finalisation latency 3057/1617 ms. Streaming-small is 1.95/3.74%, 3073/1622 ms. Latency is relative to Whisper-tiny DTW word ends, not independent boundary truth. There is no vendor streaming-policy reference for these floors.
- **Language identification:** Whisper and its reference each identified 15/15 FLEURS probes (five each English/French/German). This is information only, not multilingual coverage evidence.
- **Publisher availability:** VoxCeleb1 and VoxConverse audio are no longer available from the publisher. Their intended rows are unavailable; no unofficial re-upload is substituted.
- **Gated segmentation:** authorized segmentation parity and complete-set slot measurements are reported above. The segmentation/CAM++ slot is not a community-1 reimplementation; the community-1 comparison remains information, with its quality gaps disclosed.
- **Long-form:** full long-form rows are reserved for the 1.0 evaluation. Existing single-file observations do not establish equivalence. In particular, an Earnings-22 4474955 turbo temperature-zero comparison was 55.159% versus 58.386% reference WER: outside the 0.5-point bound, in our favour because the reference appended repeated tail text. Other decoding/pipeline settings produced different scores; they must not be combined into that row. Whole-recording Qwen reference runs exceeded the CPU run bounds.
- **Quantization and speed:** full Moonshine int8 quality is reported above; full Whisper fp32 and int8 comparisons are complete and within paired noise. The Speed section reports the measured structural-overhead and live-allocation rows, including the Nemotron allocation residual and unmeasured Qwen row; it does not establish a blanket throughput guarantee.

These results do not yet demonstrate every release target. See [evaluation conventions](eval.md) for scoring and [word timing](word-timing.md) for failure semantics.

### Community-1 pipeline (CommunityDiarizer, tier 2)

Summary, interpretation and the tuned-on-test caveat are in [diarization](diarization.md#community-1-measured-agreement). DER/JER %, no collar, overlap scored; reference is pyannote.audio 4.0.7 community-1 (CPU, offline); eight CPU threads, one recording per job.

| Meeting | DER ours / reference | JER difference |
|---|---|---|
| EN2002a | 26.66 / 21.91 | +2.79 |
| EN2002b | 27.34 / 20.74 | +5.08 |
| EN2002c | 22.84 / 17.54 | +3.78 |
| EN2002d | 31.36 / 23.17 | +6.43 |
| ES2004a | 19.80 / 18.76 | +1.08 |
| ES2004b | 14.38 / 12.61 | +1.00 |
| ES2004c | 13.17 / 12.35 | +0.18 |
| ES2004d | 20.19 / 18.67 | +0.79 |
| IS1009a | 22.70 / 19.65 | +2.21 |
| IS1009b | 16.41 / 15.19 | +0.66 |
| IS1009c | 14.80 / 10.05 | +2.93 |
| IS1009d | 23.70 / 16.45 | +2.36 |
| TS3003a | 19.06 / 17.14 | +4.66 |
| TS3003b | 12.14 / 11.97 | -1.81 |
| TS3003c | 13.44 / 13.34 | -0.54 |
| TS3003d | 21.44 / 21.08 | -0.49 |
| R8001_M8004_MS801 | 22.47 / 22.02 | +0.48 |
| R8003_M8001_MS801 | 17.47 / 17.38 | +0.11 |
| R8007_M8010_MS803 | 34.67 / 34.24 | +0.58 |
| R8007_M8011_MS806 | 20.17 / 20.17 | +0.22 |
| R8008_M8013_MS807 | 20.68 / 28.46 | -8.30 |
| R8009_M8018_MS809 | 12.31 / 11.97 | -0.41 |
| R8009_M8019_MS810 | 8.72 / 8.72 | +0.01 |
| R8009_M8020_MS810 | 7.73 / 7.84 | -0.19 |

