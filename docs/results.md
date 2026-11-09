# Measured results and limits

These are fixed-slice correctness results, not full-corpus or cross-hardware guarantees. Lower is better for all errors below. Scores are percentages unless otherwise stated. All recognition rows use float32. Models and publisher revisions are pinned in the [catalog](models.md); see [notices](../THIRD-PARTY-NOTICES.md) for attribution. No speed comparison with a native implementation is made here.

**CI convention:** intervals below are 95% paired bootstrap intervals for **our score minus the reference score**, in percentage points, not intervals on our absolute score. The available evidence does not supply absolute-score intervals. Recognition uses 1,000 utterance resamples, seed 1729, and the same `en/1` scoring normalizer on both sides. “Within noise” means the difference interval includes zero. With no computable CI, the comparison uses a 0.5-point bound. Slices must not be interpreted as full-test results.

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

- **Files:** Whisper large-v3-turbo remains the default. Its test-other WER is 5.758% in the file-model comparison; Qwen3-ASR 1.7B is 4.131%, a turbo-minus-Qwen difference of +1.63 points (paired CI [1.02, 2.31]). Qwen 1.7B is available by name for accuracy. The fixed-slice CPU comparison favoured turbo's resource requirements; the warm speed rows above do not include Qwen 1.7B.
- **Live:** Moonshine streaming-tiny is the default. On 20-file clean/other streaming slices, its WER floor is 1.70/8.33%, with mean finalisation latency 3057/1617 ms. Streaming-small is 1.95/3.74%, 3073/1622 ms. Latency is relative to Whisper-tiny DTW word ends, not independent boundary truth. There is no vendor streaming-policy reference for these floors.
- **Language identification:** Whisper and its reference each identified 15/15 FLEURS probes (five each English/French/German). This is information only, not multilingual coverage evidence.
- **Publisher availability:** VoxCeleb1 and VoxConverse audio are no longer available from the publisher. Their intended rows are unavailable; no unofficial re-upload is substituted.
- **Gated segmentation:** authorized local segmentation-3.0 inference and overlap-aware clustering are supported, as above. The network entry point still refuses inference. Community-1 reference rows are information only; it has no native pipeline adapter.
- **Long-form:** full long-form rows are reserved for the 1.0 evaluation. Existing single-file observations do not establish equivalence. In particular, an Earnings-22 4474955 turbo temperature-zero comparison was 55.159% versus 58.386% reference WER: outside the 0.5-point bound, in our favour because the reference appended repeated tail text. Other decoding/pipeline settings produced different scores; they must not be combined into that row. Whole-recording Qwen reference runs exceeded the CPU run bounds.
- **Quantization and speed:** int8 quality is pending the 1.0 evaluation. The fp32 speed rows above cover the stated workloads only; no int8 speed or universal constant-memory guarantee is made here.

These results do not yet demonstrate every release target. See [evaluation conventions](eval.md) for scoring and [word timing](word-timing.md) for failure semantics.
