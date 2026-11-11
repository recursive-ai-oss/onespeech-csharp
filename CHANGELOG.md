# Changelog

All notable changes are documented here, following Keep a Changelog.

## [Unreleased]

### Added
- `CommunityDiarizer`: the published pyannote community-1 pipeline (powerset segmentation, WeSpeaker ResNet34 embeddings, the published PLDA, VBx, count-based reconstruction) running natively, overlap-aware, loaded from an authorized local snapshot of the gated CC-BY-4.0 models. Tier 2: never a default, never redistributed. See docs/diarization.md.
- `ERes2NetV2Embedder`: the ERes2NetV2 Chinese speaker embedder (192 dimensions), available by name (`iic/speech_eres2netv2_sv_zh-cn_16k-common`); never a default.
- Nemotron 3.5 ASR streaming supports `WithPhrases`: token-level shallow fusion over the same phrase trie as Parakeet, offline, through `TranscribeStream` and in live sessions. `Capabilities.Biasing` is now `true` for this model.

### Changed
- **Word and text spacing is the same for every family.** Whisper no longer leaves its leading space on `Word.Text`, `Segment.Text`, `Transcript.Text` or turn text, so cue text in SRT/VTT/ASS/TTML matches Moonshine's. Words and segments carry no leading or trailing whitespace; a space is inserted between words only where the script uses spaces (none between Han, kana, Thai, Lao, Myanmar or Khmer text). Transcript JSON written by 1.0.0 with leading spaces still reads unchanged.
- Nemotron live sessions decode a chunk as soon as the audio it needs has arrived, instead of on the next fixed 80 ms-grid tick, so a Final word is published up to one grid step earlier. Recognized text is unchanged.

### Fixed
- `onespeech identify --help` described a default threshold of 0.3; the default is the development-fitted operating point (0.493 English, 0.723 Chinese), as the docs say.
- Live sessions with online clustering no longer emit adjacent `SpeakerTurnEvent`s for the same speaker inside one utterance; the windows are merged before the turn is emitted. A speaker who continues across two utterances still gets one turn per utterance (emitted events never change).

## [1.0.0] - 2026-11-10

First release. Measured results and limits: [docs/results.md](docs/results.md).

### Highlights
- **Transcription** with four model families behind one seam: Whisper (tiny to large-v3 and large-v3-turbo, the default), Qwen3-ASR 0.6B and 1.7B, FastConformer (Parakeet-TDT-0.6B-v3 and Nemotron 3.5 ASR streaming, by name), and Moonshine (the live default). VAD-first long-file chunking in constant memory, language identification, optional mixed-int8 variants.
- **Word timing** from the best available method in a fixed order (native, Qwen3-ForcedAligner, omniASR-CTC, DTW), recorded per word; public `Speech.Align` for any text and audio.
- **Diarization**: a tier-1 default of Silero VAD, CAM++ embeddings and spectral/AHC clustering with resegmentation, merged into the transcript. The default does **not** detect overlapping speech. Tier-2 upgrades, loaded by name from the publisher: Nemotron 3 Diarization (overlap-aware, up to eight speakers) and the gated pyannote segmentation-3.0 slot.
- **Speaker recognition**: embeddings (CAM++, ECAPA-TDNN), enrolment, identification and verification with calibrated thresholds; in-memory only, never persisted.
- **Live sessions** with one event model (`Partial`, `Final`, `Endpoint`, `SpeakerTurn`) on one audio clock; true streaming partials on Moonshine and Nemotron 3.5 ASR, utterance mode for the other families.
- **Contextual biasing** through `WithPhrases` where the model supports it; other models refuse with a typed error.
- **Packages**: `RecursiveAI.OneSpeech`, `.Media` (containers, SRT/VTT/ASS/TTML, caption muxing), `.Cli` (the `onespeech` tool), and offline weight packages `.Models.CamPlusEnglish`, `.Models.MoonshineTiny`, `.Models.MoonshineBase`, `.Models.MoonshineStreamingTiny`. NativeAOT and trim clean; no native code.
- **Models**: tier-1 (MIT, Apache-2.0) models download on first use; tier-2 (CC-BY-4.0, OpenMDW-1.1, gated) models load only by name from the publisher's files and are never redistributed.

Changes since the last preview follow.

### Added
- The gated pyannote segmentation-3.0 model in the segmentation slot, with an overlap-aware diarizer over it (`PyannoteDiarizer`), loaded from publisher files the user is entitled to; never a default.
- Mixed-int8 variants of Whisper large-v3-turbo, Whisper small and Moonshine streaming-tiny, selected by model ID (`-int8`) and converted locally from the same pinned weights; defaults remain fp32.
- Contextual phrase biasing through `SpeechRecognizer.WithPhrases`: Whisper prompt conditioning, Qwen3-ASR context, and shallow fusion over a phrase trie for Parakeet and custom CTC/RNN-T/TDT models. Families that cannot bias refuse with `CapabilityNotSupportedException`.
- `Speech.Enroll`, `Speech.Identify`, `Speech.Verify` and `Speech.ClearSpeakers`, with enrolled names on transcript speakers.
- `Speech.PrefetchAsync(language, model)` fetches every model a call would need.
- Path-input `*Async` overloads for every verb, and stream input for `TranscribeStream` and `DetectLanguageAsync`.
- Uniform `Load(ResolvedModel, ...)` and `LoadAsync(store, model, ...)` on every downloadable family.
- `ModelUnavailableException`, and `ModelId` on model errors.
- `TranscriptJsonConverter`, so plain System.Text.Json writes the documented schema-1 shape; value equality for `Transcript`, `Segment`, `Turn` and `ModelInfo`.

### Changed
- Requires RecursiveAI.OneModel 1.2.0.
- Faster CAM++ feature extraction with identical output.

### Fixed
- `Transcript.ToJson` and `FromJson` handle recordings up to 24 hours (documented limits, checked before allocating).
- One language-tag rule for every entry point; unsupported languages are refused, and `Align` no longer returns untimed words when no aligner applies.
- The role base classes implement `IDisposable`. Wrappers and sessions borrow, and never dispose, the models they use.
- `CreateLiveSession` accepts any `VoiceActivityDetector`.
- Consistent optional `language` and `cancellationToken` parameters, with the token last everywhere; `MediaInput.TranscribeStreamAsync` is renamed `TranscribeStream`.
- `WhisperOptions.Progress` is `IProgress<TimeSpan>`; `MoonshineOptions.IntraOpThreads` is renamed `Threads`; thread counts default to 1.
- Invalid in-memory PCM throws `ArgumentException`.
- The Silero VAD provenance ID is the catalog ID `silero/vad-v6.2.3`.

## [0.4.0-preview.1]

### Added
- Qwen3-ASR 0.6B and 1.7B recognition, language identification and context biasing, and Qwen3-ForcedAligner word timing.
- CTC forced alignment with omniASR-CTC-300M and public `Speech.Align` for any text and audio.
- Moonshine recognition with true streaming partials, incremental decoding with cached attention, and live sessions that keep up in real time on one core with no steady-state allocation in OneSpeech's own code per audio chunk.
- Online speaker clustering for live sessions.
- Parakeet-TDT-0.6B-v3 and Nemotron 3 Diarization by name, loaded from the publisher's files and converted locally. They are never defaults and are never redistributed. Nemotron 3 Diarization detects overlapping speech.
- Gated publisher models: a user token is sent only to the publisher's own file URLs, with typed errors for missing tokens, rejected tokens and gates not yet accepted.
- Custom CTC and RNN-T/TDT models of a known shape, decoded by the public `Decoding` primitives with native word timing.
- NeMo mel features.
- `RecursiveAI.OneSpeech.Media`: audio from any container OneMedia opens, caption export (`ToSrt`, `ToVtt`, `ToAss`, `ToTtml`) and caption muxing.
- Whisper large-v3.

### Changed
- Requires RecursiveAI.OneModel 1.2.0-preview.2.
- Qwen3 BF16 graphs are converted only when requested.

## [0.2.0-preview.1]

### Added
- Live sessions with one event model (`PartialEvent`, `FinalEvent`, `EndpointEvent`, `SpeakerTurnEvent`): finals are immutable, partials carry a revision, and all events share one audio clock. `Speech.Listen` and `Speech.TranscribeStream`; utterance mode (voice-activity endpointed finals) for offline recognizers.
- Chinese speech uses the Chinese CAM++ speaker embedder for diarization; other languages use the English one.
- A diarization speech-activity profile fitted on development meetings.

### Fixed
- Transcript JSON writes times exactly at 100-ns resolution instead of rounding through binary floating point.

## [0.1.0-preview.1]

### Added
- Shared speech contracts, the `Transcript` document model and its versioned, bounded JSON form.
- `Speech.Transcribe`, `TranscribeAsync` and `DetectLanguage`: transcription of WAV, MP3, FLAC, Ogg, M4A/3GP/ADTS and AMR files, returning words with timings, confidence and timing method, segments, language and speaker turns.
- Whisper recognition (tiny, base, small, large-v3-turbo) with greedy and beam decoding, temperature fallback, VAD-first long-form chunking, cross-attention word timing and language identification.
- Embedded Silero VAD v6.2.3.
- Speaker embeddings (CAM++ English and Chinese, ECAPA-TDNN), an in-memory speaker registry, and clustering diarization (spectral and agglomerative) merged into transcripts, including channel diarization for separated-party stereo. The clustering diarizer does not detect overlapping speech.
- A pinned model catalog with verified download, local conversion on first use, a per-user cache, the `ModelSource` hook and `onespeech models pull`.

### Known limitations
- Whisper large-v3 was refused with a typed error in this release (fixed in 0.4.0-preview.1).
