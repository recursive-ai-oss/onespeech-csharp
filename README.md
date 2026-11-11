# Speech recognition for .NET

OneSpeech transcribes audio, times words, labels speakers and supports live sessions in managed .NET 10. Open-weight models run on OneModel; OneAudio decodes and resamples audio. No native inference runtime is required.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

See [measured results and limitations](docs/results.md) before choosing a model. Licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

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

`Transcribe` returns a timed, diarized, language-tagged `Transcript`: inspect `Words`, `Segments`, `Turns` and `Models`. Failed word timing is **null**, never invented. A speaker label is anonymous and local to the document, not a person's identity. `TranscribeStream` yields final segments in bounded chunks; it does **not** perform whole-file speaker assignment. Standalone `Align` accepts at most 60 seconds. All verbs take an optional final `cancellationToken`; path and encoded-stream I/O have `*Async` forms. CPU inference remains synchronous within those calls. Language tags are case-insensitive: `en-US`, `EN`, `eng` and `English` resolve to `en`; unsupported hints throw `CapabilityNotSupportedException`.

`Speech.Listen` consumes caller-supplied mono 16 kHz PCM as an `IAsyncEnumerable<AudioBuffer>`. It emits replacing partials, immutable finals, endpoints and speaker turns on one audio clock. It does not capture a microphone. See [live sessions](docs/live-sessions.md).

For speaker recognition, use `Speech.Enroll("alice", "enrollment.wav")`, `Speech.Identify("query.wav")` and `Speech.Verify("first.wav", "second.wav")`; encoded streams have `*Async` twins. Enrollments live only in process memory: `Speech.ClearSpeakers()` removes them and releases speaker models. File transcription adds matching enrolled names to speaker labels without changing their document-local IDs; unknown clusters keep anonymous labels. Obtain consent and check local biometric law before using these verbs. See [speaker recognition](docs/speaker-recognition.md) for calibrated thresholds, limitations and the low-level registry.

### Overlapping speech

**The default clustering diarizer does not detect overlapping speech.** For an overlap-aware upgrade (up to eight speakers), load the named tier-2 diarizer in one call: `await NemotronDiarizer.LoadAsync()`, then pass it to your transcription pipeline. This does not separate simultaneous voices. See the [complete replacement example](docs/diarization.md#voice-clustering-and-overlap-limitation) and [Nemotron guide](docs/nemotron-diarization.md). It is never downloaded as a default.

## First-use downloads and offline preparation

The `model` argument selects a role, **not the entire pipeline**. Transcription also needs a speaker embedder and, without native timings, companion aligners. Even Whisper tiny can require several GB on first use. Silero VAD is embedded. See the [per-verb download table](docs/model-downloads.md) for exact IDs, sizes and conditional dependencies.

`await Speech.PrefetchAsync(language: "en", model: "openai/whisper-tiny");` prepares the complete companion set, including timing fallbacks, without running inference. For live use, pass `model: "moonshine-ai/moonshine-streaming-tiny"`. Every downloadable family exposes `Load(ResolvedModel, …)` and `LoadAsync(store, model, …)`; [composition examples](docs/loading.md) use documented role types.

## Contextual biasing

Biasing is a low-level request; `Speech.*` stays unchanged. A loaded recognizer
returns a borrowed wrapper with its own immutable phrase list:

```csharp
using var model = OneSpeech.Whisper.WhisperRecognizer.Load(modelDirectory);
SpeechRecognizer biased = model.WithPhrases([new Phrase("Acme Photonics"), new Phrase("Marisol")]);
Transcript result = biased.Transcribe(audio, "en", cancellationToken);
// The same biased recognizer can be passed to VadTranscriptionPipeline.
```

Keep the owning model alive until the wrappers and their sessions are finished.
Wrappers do not own or dispose it; the model's concurrency limits still apply.
Calling `WithPhrases` again replaces the list; an empty list returns the original
unbiased recognizer. The original recognizer is never modified.

Whisper conditions each window with ordinary prompt tokens. Qwen3-ASR appends
phrases to its existing system context. These prompt methods refuse explicit
numeric weights rather than pretending to apply a score bonus. Parakeet TDT
uses token-level shallow fusion; `new Phrase("Acme", 0.5)` supplies a natural-log
bonus per token (null defaults to 1). Custom CTC/transducer graphs use the same
fusion after the caller supplies `TokenVocabulary.WithPhraseEncoder` for their
exact tokenizer. CTC biasing uses prefix beam search, even if the unbiased
recognizer is greedy. Moonshine does not support biasing: a nonempty request
throws `CapabilityNotSupportedException`, as declared by `Capabilities.Biasing`.

Biasing is not a spelling replacement or a guarantee that a term will be spoken.
It can introduce false alarms. Whisper prompt conditioning in particular can
substantially increase overall errors and repeat absent terms even when recall
of requested terms improves; it is not a safe default. Check both phrase recall and overall WER, with
absent-phrase controls, on your audio. See [contextual scoring and bounds](docs/contextual-biasing.md)
for prompt budgets, weights, decoder semantics and the repeatable evaluation.

## Packages

| Package | Purpose |
|---|---|
| `RecursiveAI.OneSpeech` | Recognition, alignment, diarization, speakers, live sessions, algorithms and JSON |
| `RecursiveAI.OneSpeech.Media` | Container input, SRT/VTT/ASS/TTML and caption muxing using OneMedia |
| `RecursiveAI.OneSpeech.Models.CamPlusEnglish` | Optional offline-at-restore CAM++ English weights |
| `RecursiveAI.OneSpeech.Models.MoonshineTiny`, `.MoonshineBase`, `.MoonshineStreamingTiny` | Optional offline-at-restore Moonshine weights |
| `RecursiveAI.OneSpeech.Cli` | The `onespeech` dotnet tool |

Models not supplied by a package download on first use; CPU runtime graphs are built and verified in the per-user cache. An offline model package covers only that model, not every role used by `Transcribe`. See [cache, pre-fetch and offline use](docs/models.md) and [CLI and model packages](docs/cli-and-model-packages.md).

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
| Segmentation | pyannote segmentation-3.0 / community-1 | 2 / MIT gated / CC-BY-4.0 gated | Local segmentation-3.0 + our overlap-aware clustering; community-1 reference only |
| Speaker embeddings | CAM++ English/Chinese; ECAPA-TDNN; ERes2NetV2 Chinese | 1 / Apache-2.0 | CAM++ English default; Chinese for Chinese speech |

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

Set `ONESPEECH_ARTIFACTS` to a directory outside the checkout. Run `dotnet restore`, then `dotnet build --no-restore`. Run `dotnet run --project tests/DocsSamples --no-build -- .` to exercise the no-download document examples and check local documentation links. Model-backed examples are compile-checked without downloading weights; run them with your own consented audio and fetched models.
