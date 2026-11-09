# Measured results and limits

The release revalidation table distinguishes full-split measurements from the earlier fixed-slice evidence below. The release matrix is still incomplete; these results are not a cross-hardware guarantee. Lower is better for all errors below. Scores are percentages unless otherwise stated. Unless noted, recognition quality rows use float32. Models and publisher revisions are pinned in the [catalog](models.md); see [notices](../THIRD-PARTY-NOTICES.md) for attribution. No speed comparison with a native implementation is made here.

**CI convention:** the release table labels absolute-score and paired-difference 95% intervals separately. The older slice tables give only intervals for **our score minus the reference score**, in percentage points. Recognition uses 1,000 utterance resamples, seed 1729, and the same `en/1` scoring normalizer on both sides. “Within noise” means the difference interval includes zero. With no computable CI, the comparison uses a 0.5-point bound. Slices must not be interpreted as full-test results.

## Release revalidation (2026-11-08, incomplete)

The inference measurements started from core source `dd63517` through the public recognizer, live-session and embedding APIs. Later reader hardening changes invalid-input handling, not valid-model inference. Moonshine uses its default greedy decoding, 6.5 generated tokens per audio second, no biasing, and two CPU threads. The reference uses the pinned publisher weights, Transformers 5.22.2, float32, eager attention and deterministic generation. Speaker extraction uses eight threads on each side. No previously saved hypotheses or embeddings were substituted for the new inference runs.

| Model / task | Dataset | Metric | Ours [absolute 95% CI] | Reference / version | Paired difference CI | Verdict |
|---|---|---|---|---|---|---|
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

**Remaining release evidence:** full Whisper small/turbo and Qwen3-ASR-1.7B/Parakeet recognition; full Whisper int8 recognition; five-recording long-form comparisons; timing revalidation; full AMI test and all eight AliMeeting Eval diarization comparisons; and Nemotron ASR streaming revalidation. The fixed shortest-duration Earnings-22 selection is `4474955`, `4483046`, `4468919`, `4479524`, `4481221` (875.020, 1193.180, 1214.487, 1226.213, 1236.904 seconds). Selection alone is not a measured long-form row. Earlier misses below remain visible and are not superseded by unmeasured rows.

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

## Model choices and remaining evidence

- **Files:** Whisper large-v3-turbo remains the default. Its test-other WER is 5.758% in the file-model comparison; Qwen3-ASR 1.7B is 4.131%, a turbo-minus-Qwen difference of +1.63 points (paired CI [1.02, 2.31]). Qwen 1.7B is available by name for accuracy. The fixed-slice CPU comparison favoured turbo's resource requirements; publishable benchmark rows are still pending.
- **Live:** Moonshine streaming-tiny is the default. On 20-file clean/other streaming slices, its WER floor is 1.70/8.33%, with mean finalisation latency 3057/1617 ms. Streaming-small is 1.95/3.74%, 3073/1622 ms. Latency is relative to Whisper-tiny DTW word ends, not independent boundary truth. There is no vendor streaming-policy reference for these floors.
- **Language identification:** Whisper and its reference each identified 15/15 FLEURS probes (five each English/French/German). This is information only, not multilingual coverage evidence.
- **Publisher availability:** VoxCeleb1 and VoxConverse audio are no longer available from the publisher. Their intended rows are unavailable; no unofficial re-upload is substituted.
- **Gated segmentation:** pyannote inference and comparison rows are pending authorized access. Catalog presence is not evidence of a functioning segmentation inference implementation.
- **Long-form:** full long-form rows are reserved for the 1.0 evaluation. Existing single-file observations do not establish equivalence. In particular, an Earnings-22 4474955 turbo temperature-zero comparison was 55.159% versus 58.386% reference WER: outside the 0.5-point bound, in our favour because the reference appended repeated tail text. Other decoding/pipeline settings produced different scores; they must not be combined into that row. Whole-recording Qwen reference runs exceeded the CPU run bounds.
- **Quantization and speed:** int8 quality and publishable benchmark rows are pending the 1.0 evaluation. No int8 quality, structural-overhead or cross-platform throughput guarantee is made here.

These results do not yet demonstrate every release target. See [evaluation conventions](eval.md) for scoring and [word timing](word-timing.md) for failure semantics.
