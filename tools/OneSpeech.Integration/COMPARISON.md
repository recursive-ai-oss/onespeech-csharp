# File-transcription comparison

The integration executable has resumable, atomic per-recording comparison modes:

```
dotnet run --project tools/OneSpeech.Integration -- comparison-manifest CORPUS_SPLIT manifest.jsonl
dotnet run --project tools/OneSpeech.Integration -- comparison-earnings EARNINGS_ROOT earnings.jsonl
dotnet run --project tools/OneSpeech.Integration -- comparison MANIFEST OUTPUT MODEL_ID BEGIN COUNT
dotnet run --project tools/OneSpeech.Integration -- comparison-greedy MANIFEST OUTPUT WHISPER_DIRECTORY BEGIN COUNT
dotnet run --project tools/OneSpeech.Integration -- comparison-clips INPUT_AUDIO OUTPUT_DIRECTORY
dotnet run --project tools/OneSpeech.Integration -- comparison-speed MANIFEST OUTPUT MODEL_ID BEGIN COUNT
```

`comparison` calls `Speech.Transcribe` with an English hint and keeps its actual timing and diarization policy. `comparison-greedy` measures Whisper fp32 greedy recognition, temperature zero, eight graph threads, without timing, for comparison with existing greedy quality rows. Manifest rows have `id`, `audio`, and `text` strings. Generated LibriSpeech manifests select the first 200 ordinal recording IDs. Earnings selection uses 4474955 and the first ordinal all-numeric ID, before scoring.

Each recording produces the original Transcript JSON and a measurement sidecar (elapsed seconds, process-lifetime peak RSS in bytes, effective processor count). Existing recording outputs are skipped. Completed records are exported as `hypotheses.jsonl` and `truth.jsonl` for `onespeech-eval score --normalizer en --resamples 1000 --seed 1729`. A killed recording has no reusable result; previously completed recordings survive.

`comparison-clips` writes the first 60 and 600 seconds as mono 16 kHz WAV, using OneAudio. It requires at least 600 seconds. `comparison-speed` composes the same recognition, timing and English diarization parts as the file helper, but explicitly sets graph threads from `DOTNET_PROCESSOR_COUNT`. Silero remains single-threaded. This is necessary because the static helper does not expose graph thread settings. Qwen recognition is followed by its forced aligner before generic CTC fallback. Measurements include model resolution/loading, decoding, timing and diarization, with no warm-up excluded. Run each size/model/thread combination in a fresh process for a separate peak RSS. These are indicative shared-machine measurements, not isolated benchmark claims.

For pooled timing summaries, put MFA truth in `truth.jsonl`, candidate hypotheses in `ctc.jsonl`, `dtw.jsonl`, `oracle.jsonl`, then run `comparison-summary DIRECTORY`. Those three filenames are labels inherited from the timing-floor summarizer, not declarations of the candidates' timing methods. The summary includes all matched words, unmatched truth, untimed hypotheses, and a literal same-word intersection across all candidates.
