# Changelog

All notable changes are documented here, following Keep a Changelog.

## [Unreleased]

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
