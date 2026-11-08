# Shared contracts

`AudioBuffer` views interleaved float PCM without copying it. Mono is one channel; multiple channels preserve separated-party audio for later channel diarization. Sample rates are positive Hz and buffers contain whole frames; processing implementations document rate/channel support. Float samples conventionally lie in [-1,1]; the buffer does not scan, clip or resample them. Callers must not mutate borrowed samples while processing.

`Transcript`, `Word`, `Segment`, `Turn`, `SpeakerLabel`, `ModelInfo`, and `Capabilities` are immutable records with immutable collection values. Text is explicit rather than guessed from punctuation or token spacing. `SpeakerLabel.Unassigned` is distinct from a null (unprocessed) speaker. Failed timings remain null.

Roles process in-memory audio synchronously and accept cancellation tokens; recognition, alignment, VAD and diarization entry points default them to CancellationToken.None. Recognizers and pipelines also expose synchronous bounded segment enumeration. Asynchronous APIs are reserved for I/O: the `ModelSource` stream hook and the `Speech.TranscribeAsync` input-stream entry point. Models declare support through `Capabilities`; empty languages means no support declared, not unrestricted support. Unsupported capabilities produce a typed error. Typed family options, implementations, live events, and loading policy will be added separately.

`CatalogEntry` and `CatalogFile` describe pinned bytes and their provenance, not loading authorization. `ModelSource` returns caller-owned readable streams; the consuming loader must enforce tier/gate rules, relative-path safety, size, digest, HTTPS and revision policies even for custom sources. `CatalogEntry.FromJson` uses the transcript reader's structural bounds and generated metadata; it is data deserialization only, not semantic catalog verification.

`Speech` forwards to the VAD transcription pipeline through an explicit internal recognizer resolver. Without a configured resolver it fails with `FeatureUnavailableException` before opening input. No recognition or timing is simulated. See [audio input](audio-input.md) and [Silero VAD](vad.md) for decoding, chunking, audio-clock and lifetime behaviour.

`SpeakerEmbedder` adds synchronous, caller-buffer extraction of speaker vectors, with explicit dimension,
mono sample rate, model provenance and disposal. `InMemorySpeakerRegistry` preserves the abstract
`SpeakerRegistry` surface and adds multi-utterance enrollment; its threshold is explicit. See
[speaker identity and scores](speakers.md) for averaging, normalization, calibration and privacy.

## Language hints and argument failures

All entry points share one language rule: trim whitespace, match case-insensitively, accept unambiguous English language names and ISO three-letter aliases, and reduce region/script variants to the model's language-only tag. For example, `en-US`, `EN`, `English`, `eng` and `eng_Latn` become `en`; `zh-Hant-TW` and `Chinese` become `zh`. The output language uses the normalized tag. A hint unknown to the library or unsupported by the selected model throws `CapabilityNotSupportedException`, even on silence. `DetectLanguage` refuses any hint because it determines the language itself. Ordered `Align` refuses an uncovered language; no-model-run is not successful alignment. Speech absent or unalignable text **after an aligner runs** still legitimately has null word timing.

Invalid caller-supplied PCM (rate, channel shape, nonfinite samples) throws `ArgumentException`; decoding corrupt or unsupported **encoded** input throws `UnsupportedAudioException`, with `SourceName` and the path in the message for files. Resample/downmix in-memory PCM with OneAudio before calling a role. `ModelNotFoundException` means an unknown catalog ID/default; `ModelUnavailableException` means a known model/file is absent from the selected publisher or offline source. Model errors carry `ModelId` where available, and gate errors name the publisher repository.

Roles implement `IDisposable`. They release their owned graph/state only; wrappers, live sessions and composed pipelines borrow their collaborators. Dispose a session before disposing its recognizer/detector. All public `Speech` methods place optional `CancellationToken cancellationToken = default` last, after optional `language` and `model`. Awaitable single results use `Async`; segment/event sequences retain `TranscribeStream`/`Listen` in both core and Media. In-memory CPU work is synchronous; async path/stream methods await acquisition/input copying and run inference on the caller's continuation, not `Task.Run`.

See [first-use download dependencies](model-downloads.md) and [role loading/composition](loading.md). `Speech.PrefetchAsync` prepares all possible companions for the selected recognizer without inference; it may fetch more than one particular input needs.
