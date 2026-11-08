# Changelog

All notable changes are documented here, following Keep a Changelog.

## [Unreleased]

## [0.1.0-preview.1]

### Added
- Shared speech contracts, the `Transcript` document model and its versioned, bounded JSON form.
- `Speech.Transcribe`, `TranscribeAsync` and `DetectLanguage`: transcription of WAV, MP3, FLAC, Ogg, M4A/3GP/ADTS and AMR files, returning words with timings, confidence and timing method, segments, language and speaker turns.
- Whisper recognition (tiny, base, small, large-v3-turbo) with greedy and beam decoding, temperature fallback, VAD-first long-form chunking, cross-attention word timing and language identification.
- Embedded Silero VAD v6.2.3.
- Speaker embeddings (CAM++ English and Chinese, ECAPA-TDNN), an in-memory speaker registry, and clustering diarization (spectral and agglomerative) merged into transcripts, including channel diarization for separated-party stereo. The clustering diarizer does not detect overlapping speech.
- A pinned model catalog with verified download, local conversion on first use, a per-user cache, the `ModelSource` hook and `onespeech models pull`.

### Known limitations
- Whisper large-v3 is refused with a typed error: its decoder exceeds the current model-file size limit. Use large-v3-turbo.
