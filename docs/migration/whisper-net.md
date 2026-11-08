# From Whisper.net

This is a concept map, not an API-compatible shim. It uses the [published package documentation](https://www.nuget.org/packages/Whisper.net/), consulted 2026-11-08; no implementation source is needed.

| Familiar concept | OneSpeech equivalent |
|---|---|
| Factory loading a model path, then a processor builder | A catalog model ID for `Speech.Transcribe`, or a disposable low-level family recognizer |
| Asynchronously enumerating processor segments | `Speech.TranscribeStream` yields immutable `Segment` values |
| Language builder setting | Optional `language` scalar or typed family options |
| Segment timestamps | Segment boundaries plus per-word timing provenance; missing times are null |
| Runtime/model installation | Core managed package plus verified catalog weights; no native runtime package |

Start with the [README example](../../README.md#transcribe-stream-and-align). Do not pass a GGML model file to OneSpeech: use its supported publisher files and local runtime conversion. Similar model names do not imply identical decoding policy, timestamps or long-file segmentation. Check the [results](../results.md) for the exact slices compared.

For complete diarization, use `Transcribe`; segment streaming does not run whole-file speaker clustering. See [long files](../long-files.md), and keep cancellation and enumeration disposal in the application lifetime. OneSpeech is not affiliated with Whisper.net.
