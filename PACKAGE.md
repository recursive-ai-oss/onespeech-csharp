# Speech recognition for .NET

OneSpeech transcribes audio, times words, labels speakers and runs live sessions in managed .NET 10. Open-weight models (Whisper, Qwen3-ASR, FastConformer, Moonshine, Silero VAD, CAM++) run on OneModel; OneAudio decodes and resamples audio. No native inference runtime is required, and the packages are NativeAOT and trim clean.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

```csharp
using OneSpeech;

Transcript transcript = Speech.Transcribe("meeting.m4a");
foreach (Turn turn in transcript.Turns)
    Console.WriteLine($"{turn.Speaker}: {turn.Text}");
```

Models download on first use into a per-user cache, verified by SHA-256. The default clustering diarizer does not detect overlapping speech; an overlap-aware tier-2 diarizer is available by name. Read the [measured results and limits](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/docs/results.md) before choosing a model.

[Source and documentation](https://github.com/recursive-ai-oss/onespeech-csharp/tree/main) · [Project](https://recursiveai.org/onespeech/) · [Licences](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/LICENSE-MIT) · [Third-party notices](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/THIRD-PARTY-NOTICES.md)

## Container audio and subtitles

The optional `RecursiveAI.OneSpeech.Media` package adds OneMedia container audio input, `Transcript` subtitle exports (SRT, WebVTT, ASS and TTML), cue interchange and caption muxing. [Usage and fidelity policy](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/docs/media.md).

## Command line

`dotnet tool install -g RecursiveAI.OneSpeech.Cli` provides `onespeech transcribe`, `align`, `diarize`, `enroll` and `models pull`.
