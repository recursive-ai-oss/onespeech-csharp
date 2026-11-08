# Speech recognition for .NET

OneSpeech is building transcription, word timestamps, diarization, speaker recognition and live sessions in managed .NET, without native code. Open-weight models run on OneModel, with audio input through OneAudio.

**Status: in development, no public release yet.** Whisper and Qwen3-ASR recognition, forced alignment, VAD and speaker processing run natively on OneModel. Shared contracts and bounded transcript JSON remain the document seam.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

Targets .NET 10. Licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE.

## Development

Set `ONESPEECH_ARTIFACTS` to an absolute directory outside the checkout, then run `dotnet restore`, `dotnet build -c Release --no-restore`, and `dotnet test tests/OneSpeech.Tests -c Release --no-build`. The default output directory is `onespeech-build` in the system temporary directory. `dotnet pack -c Release --no-build` creates local packages only.

See [Qwen3 recognition and alignment](docs/qwen3.md), [the transcript JSON format](docs/transcript-json.md) and [the shared contracts](docs/contracts.md).

## Containers and captions

`RecursiveAI.OneSpeech.Media` adds OneMedia container audio input, subtitle export and MP4 caption muxing. Call `OneSpeech.Media.MediaInput.Register()` once at startup; then `Speech.Transcribe("meeting.mkv")` uses the container's audio track. Transcript extensions `ToSrt`, `ToVtt`, `ToAss` and `ToTtml` use OneMedia's writers. See [container audio and captions](docs/media.md) for untimed words, fidelity reports, streams and muxing.
