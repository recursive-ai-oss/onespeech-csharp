# Speech recognition for .NET

OneSpeech is building transcription, word timestamps, diarization, speaker recognition and live sessions in managed .NET, without native code. Open-weight models will run on OneModel, with audio input through OneAudio.

**Status: in development, no release yet.** The repository currently provides shared contracts and bounded transcript JSON serialization, not a working recognizer.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

Targets .NET 10. Licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE.

## Development

Set `ONESPEECH_ARTIFACTS` to an absolute directory outside the checkout, then run `dotnet restore`, `dotnet build -c Release --no-restore`, and `dotnet test tests/OneSpeech.Tests -c Release --no-build`. The default output directory is `onespeech-build` in the system temporary directory. `dotnet pack -c Release --no-build` creates local packages only.

See [the transcript JSON format](docs/transcript-json.md) and [the shared contracts](docs/contracts.md).
