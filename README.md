# Speech recognition for .NET

OneSpeech transcribes audio, times words, labels speakers and supports live sessions in managed .NET 10. Open-weight models run on OneModel; OneAudio decodes and resamples audio. No native inference runtime is required.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

**In development; no public release yet.** See [measured results and limitations](docs/results.md) before choosing a model. Licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

## Transcribe, stream and align

```csharp
using OneSpeech;

Transcript transcript = Speech.Transcribe("meeting.m4a", language: "en");
Console.WriteLine(transcript.Text);
File.WriteAllText("meeting.json", transcript.ToJson());

await foreach (Segment segment in Speech.TranscribeStream("lecture.wav"))
    Console.WriteLine(segment.Text);

Transcript aligned = Speech.Align("short.wav", "The words that were spoken.", language: "en");
```

`Transcribe` returns a timed, diarized, language-tagged `Transcript`: inspect `Words`, `Segments`, `Turns` and `Models`. Failed word timing is **null**, never invented. A speaker label is anonymous and local to the document, not a person's identity. `TranscribeStream` yields final segments in bounded chunks; it does **not** perform whole-file speaker assignment. Standalone `Align` accepts at most 60 seconds. File calls have explicit cancellation overloads; encoded-stream I/O uses `TranscribeAsync` / `AlignAsync`.

`Speech.Listen` consumes caller-supplied mono 16 kHz PCM as an `IAsyncEnumerable<AudioBuffer>`. It emits replacing partials, immutable finals, endpoints and speaker turns on one audio clock. It does not capture a microphone. See [live sessions](docs/live-sessions.md).

For speaker recognition, use `Speech.Enroll("alice", "enrollment.wav")`, `Speech.Identify("query.wav")` and `Speech.Verify("first.wav", "second.wav")`; encoded streams have `*Async` twins. Enrollments live only in process memory: `Speech.ClearSpeakers()` removes them and releases speaker models. File transcription adds matching enrolled names to speaker labels without changing their document-local IDs; unknown clusters keep anonymous labels. Obtain consent and check local biometric law before using these verbs. See [speaker recognition](docs/speaker-recognition.md) for calibrated thresholds, limitations and the low-level registry.

### Overlapping speech

**The default clustering diarizer does not detect overlapping speech.** For an overlap-aware upgrade (up to eight speakers), load the named tier-2 diarizer in one call: `await NemotronDiarizer.LoadAsync()`, then pass it to your transcription pipeline. This does not separate simultaneous voices. See the [complete replacement example](docs/diarization.md#voice-clustering-and-overlap-limitation) and [Nemotron guide](docs/nemotron-diarization.md). It is never downloaded as a default.

## Packages

| Package | Purpose |
|---|---|
| `RecursiveAI.OneSpeech` | Recognition, alignment, diarization, speakers, live sessions, algorithms and JSON |
| `RecursiveAI.OneSpeech.Media` | Container input, SRT/VTT/ASS/TTML and caption muxing using OneMedia |
| `RecursiveAI.OneSpeech.Models.CamPlusEnglish` | Optional offline-at-restore CAM++ English weights |
| `RecursiveAI.OneSpeech.Models.MoonshineTiny`, `.MoonshineBase`, `.MoonshineStreamingTiny` | Optional offline-at-restore Moonshine weights |
| `RecursiveAI.OneSpeech.Cli` | The `onespeech` dotnet tool |

Models not supplied by a package download on first use; CPU runtime graphs are built and verified in the per-user cache. An offline model package covers only that model, not every role used by `Transcribe`. See [cache, pre-fetch and offline use](docs/models.md) and [CLI and model packages](docs/cli-and-model-packages.md). Packages are not yet on nuget.org.

## Models by role

| Role | Models | Tier / licence | Selection |
|---|---|---|---|
| File recognition | Whisper tiny/base/small/large-v3; large-v3-turbo | 1 / Apache-2.0; turbo MIT | **turbo default** |
| File recognition | Qwen3-ASR 0.6B / 1.7B | 1 / Apache-2.0 | By name; 1.7B accuracy option |
| Live / edge recognition | Moonshine English tiny/base and streaming tiny/small/medium | 1 / MIT | **streaming-tiny live default** |
| Recognition | Parakeet-TDT-0.6B-v3 | 2 / CC-BY-4.0 | By name only |
| Timing | Qwen3-ForcedAligner-0.6B; omniASR-CTC-300M | 1 / Apache-2.0 | Timing policy; see [word timing](docs/word-timing.md) |
| Voice activity | Silero VAD 6.2.3 | 1 / MIT | Embedded |
| Diarization | Silero + CAM++ + spectral/AHC clustering | 1 / MIT + Apache-2.0 | Default; no overlap detection |
| Diarization | Nemotron 3 Diarization | 2 / OpenMDW-1.1 | Explicit overlap-aware upgrade |
| Segmentation | pyannote segmentation-3.0 / community-1 | 2 / MIT gated / CC-BY-4.0 gated | Gate required; inference rows pending |
| Speaker embeddings | CAM++ English/Chinese; ECAPA-TDNN | 1 / Apache-2.0 | CAM++ English default; Chinese for Chinese speech |

Tier 1 is ungated MIT/Apache-2.0/BSD and may be redistributed and selected by default. Tier 2 loads only from pinned publisher files, never from our mirror or model packages, never by default. Local conversion stays in your cache. Licence, revision, file digests and attribution travel in `ModelInfo`. Only the listed licence-cleared Moonshine checkpoints are supported; Community License checkpoints are excluded. See [gated models](docs/tier-2-models.md) and [third-party notices](THIRD-PARTY-NOTICES.md).

## Privacy and consent

Speaker embeddings are biometric data. Obtain informed consent, limit retention and access, and check applicable law, including **Illinois BIPA** and **GDPR Article 9**, before enrollment, identification or verification. The library does not persist embeddings; applications remain responsible for audio, transcripts, logs and any copies they retain. This is not legal advice.

## Guides

- [Long files](docs/long-files.md), [live events](docs/live-sessions.md), [word timing](docs/word-timing.md)
- [Diarization](docs/diarization.md), [speaker recognition](docs/speaker-recognition.md), [biasing](docs/biasing.md)
- [Custom CTC/transducer graphs](docs/custom-models.md), [model cache](docs/models.md), [tier-2 access](docs/tier-2-models.md)
- [Transcript JSON schema v1](docs/transcript-json.md), [containers and captions](docs/media.md), [CLI](docs/cli-and-model-packages.md)
- Migration: [Whisper.net](docs/migration/whisper-net.md), [sherpa-onnx .NET](docs/migration/sherpa-onnx.md), [Azure Speech SDK](docs/migration/azure-speech.md), [WhisperX concepts](docs/migration/whisperx.md)

## Build and sample checks

Set `ONESPEECH_ARTIFACTS` to a directory outside the checkout. Run `dotnet restore`, then `dotnet build --no-restore`. The current OneModel dependency requires its local feed through `ONESPEECH_LOCAL_FEED`. Run `dotnet run --project tests/DocsSamples --no-build -- .` to exercise the no-download document examples and check local documentation links. Model-backed examples are compile-checked without downloading weights; run them with your own consented audio and fetched models.
