# Speech recognition for .NET

OneSpeech is in development; no release yet. Shared contracts and versioned transcript JSON are available in this source tree. Recognition models, file transcription, timing and speaker processing are not implemented yet.

The planned library provides managed transcription, word timestamps, diarization, speaker recognition and live sessions on .NET 10, without native code.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

[Source and documentation](https://github.com/recursive-ai-oss/onespeech-csharp/tree/main) · [Project](https://recursiveai.org/onespeech/) · [Licences](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/LICENSE-MIT)

## Container audio and subtitles

The optional `RecursiveAI.OneSpeech.Media` package adds OneMedia container audio input, `Transcript` subtitle exports (SRT, WebVTT, ASS and TTML), cue interchange and caption muxing. [Usage and fidelity policy](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/docs/media.md).
