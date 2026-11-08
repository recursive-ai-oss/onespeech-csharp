# First-use models and offline preparation

`model` selects the requested role, not an isolated inference graph. Present files are verified and reused. Nothing is downloaded on `Transcript` export. Tier-2 models must be explicitly named and always come from their publisher; prefetch never accepts a gate for you. Silero VAD (`silero/vad-v6.2.3`, 2.3 MB) is embedded in core and never needs a runtime download.

| Verb | Selected model | Companions that can be acquired |
|---|---|---|
| `Transcribe` / `TranscribeAsync` | `openai/whisper-large-v3-turbo` by default, or the named recognizer | CAM++ English or Chinese based on the requested/detected language. Without native word timing: Qwen3-ForcedAligner for its covered languages, then omniASR-CTC when words remain untimed or Qwen cannot handle the chunk; existing DTW is last fallback. |
| `TranscribeStream` (path or encoded stream) | Same recognizer default | No diarization embedder. Qwen3-ASR uses Qwen3-ForcedAligner; other families use native/DTW word output. |
| `Listen` | `moonshine-ai/moonshine-streaming-tiny` by default | CAM++ English for non-Chinese hints, Chinese for `zh`; both when no hint is supplied. Qwen3-ASR additionally uses its forced aligner. |
| `Align` / `AlignAsync` | Ordered timing by default | Qwen3-ForcedAligner for a covered hint, then omniASR-CTC if needed; without a hint only omniASR-CTC. An explicit aligner ID loads only that model. No speaker models. |
| `DetectLanguage` / `DetectLanguageAsync` | Same recognizer default as `Transcribe` | No timing or speaker companion. |
| `Enroll`, `Identify`, `Verify` and their async forms | CAM++ English by default; Chinese for `zh`, or either explicit CAM++ ID | No other downloadable model. The process-wide registry retains the selected embedder until `ClearSpeakers`. |

The timing ID is `Qwen/Qwen3-ForcedAligner-0.6B` (1,835.5 MB publisher weights, approximately 3,671 MB converted fp32). CTC is `facebook/omniASR-CTC-300M` (1,302.1 MB publisher weights, 1,302.2 MB converted). Qwen supports `zh`, `en`, `yue`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru`, `es` and needs more than 200 input samples. Standalone alignment is limited to 60 s.

English CAM++ is `iic/speech_campplus_sv_en_voxceleb_16k` (29.4 MB publisher and 29.4 MB converted); Chinese is `iic/speech_campplus_sv_zh-cn_16k-common` (28.0 MB publisher and 28.1 MB converted). Whisper tiny publisher weights are 75.6 MB, turbo 1,617.8 MB. Moonshine streaming tiny publisher weights are 176.2 MB, approximately 315 MB converted. Sizes are decimal MB, rounded; publisher and runtime files both occupy cache space. Catalog file sizes and digests are authoritative.

```csharp
await Speech.PrefetchAsync(language: "en", model: "openai/whisper-tiny",
    cancellationToken: cancellationToken);
```

`PrefetchAsync` defaults to the transcription model and prepares a **superset** of the selected recognizer's file/live needs: the recognizer, applicable Qwen timing, CTC fallback, and the chosen CAM++ embedder(s). With no language it includes both embedders and Qwen timing for eventual language detection. This deliberately includes companions a particular silent or already-timed input might not use, so a subsequent call can stay offline. Pass the live default ID to prepare `Listen`. Pass an explicit aligner or embedder ID to prefetch only that role; for a default ordered alignment, prefetch its possible aligners individually. `ModelStore.PullAsync(id)` remains the precise single-model operation. All prefetch calls are cancellable and load no inference sessions.
