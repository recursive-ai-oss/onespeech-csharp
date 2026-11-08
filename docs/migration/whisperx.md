# From WhisperX concepts

This guide uses only the [published package description](https://pypi.org/project/whisperx/), consulted 2026-11-08. It maps a workflow, not Python modules, dictionaries or an API signature.

| Familiar step | OneSpeech equivalent |
|---|---|
| ASR transcription | A `SpeechRecognizer`, or `Speech.Transcribe` for the integrated file path |
| Forced alignment for word timestamps | `Aligner` / `Speech.Align`, with nullable timing and per-word `TimingMethod` |
| Speaker diarization | A `Diarizer` producing original-clock `Turn` values |
| Assigning words to speakers | `TranscriptDiarization.Merge`, greatest covered duration with documented overlap rules |
| Saving segments/words | Immutable `Transcript` and schema-v1 JSON |

The integrated helper performs the supported workflow, but its defaults are different. The default diarizer **does not detect overlap**; use the explicit [Nemotron upgrade](../nemotron-diarization.md) if you need overlapping turns. An authorized local segmentation-3.0 snapshot can also supply [overlapping turns with our clustering](../gated-segmentation.md); community-1 has no native pipeline adapter. No noncommercial alignment weights are loaded.

Timing follows the [supported timing policy](../word-timing.md), not a language-to-aligner dictionary copied from another project. Standalone alignment currently has a 60-second limit. Words without valid timing stay untimed, and words with no covering turn are `Unassigned`; neither is silently fabricated. Inspect [timing and diarization results](../results.md) before treating labels or boundaries as ground truth.
