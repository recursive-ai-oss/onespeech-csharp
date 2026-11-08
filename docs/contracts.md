# Shared contracts

`AudioBuffer` views interleaved float PCM without copying it. Mono is one channel; multiple channels preserve separated-party audio for later channel diarization. Sample rates are positive Hz and buffers contain whole frames; processing implementations document rate/channel support. Float samples conventionally lie in [-1,1]; the buffer does not scan, clip or resample them. Callers must not mutate borrowed samples while processing.

`Transcript`, `Word`, `Segment`, `Turn`, `SpeakerLabel`, `ModelInfo`, and `Capabilities` are immutable records with immutable collection values. Text is explicit rather than guessed from punctuation or token spacing. `SpeakerLabel.Unassigned` is distinct from a null (unprocessed) speaker. Failed timings remain null.

Roles process in-memory audio synchronously and require an explicit cancellation token. Recognizers and pipelines also expose synchronous bounded segment enumeration. Asynchronous APIs are reserved for I/O: the `ModelSource` stream hook and the `Speech.TranscribeAsync` input-stream entry point. Models declare support through `Capabilities`; empty languages means no support declared, not unrestricted support. Unsupported capabilities produce a typed error. Typed family options, implementations, live events, and loading policy will be added separately.

`CatalogEntry` and `CatalogFile` describe pinned bytes and their provenance, not loading authorization. `ModelSource` returns caller-owned readable streams; the consuming loader must enforce tier/gate rules, relative-path safety, size, digest, HTTPS and revision policies even for custom sources. `CatalogEntry.FromJson` uses the transcript reader's structural bounds and generated metadata; it is data deserialization only, not semantic catalog verification.

`Speech` forwards to the VAD transcription pipeline through an explicit internal recognizer resolver. Without a configured resolver it fails with `FeatureUnavailableException` before opening input. No recognition or timing is simulated. See [audio input](audio-input.md) and [Silero VAD](vad.md) for decoding, chunking, audio-clock and lifetime behaviour.

`SpeakerEmbedder` adds synchronous, caller-buffer extraction of speaker vectors, with explicit dimension,
mono sample rate, model provenance and disposal. `InMemorySpeakerRegistry` preserves the abstract
`SpeakerRegistry` surface and adds multi-utterance enrollment; its threshold is explicit. See
[speaker identity and scores](speakers.md) for averaging, normalization, calibration and privacy.
