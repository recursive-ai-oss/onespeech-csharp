# Transcribing files

Speech recognition for .NET. OneSpeech is unaffiliated with other products or companies named OneSpeech.

```csharp
using OneSpeech;

Transcript transcript = Speech.Transcribe("voicemail.m4a");
Console.WriteLine(transcript.Text);
File.WriteAllText("voicemail.json", transcript.ToJson());

Transcript small = Speech.Transcribe("voicemail.wav", language: "en", model: "openai/whisper-tiny");
string language = Speech.DetectLanguage("voicemail.mp3");
```

The provisional file default is Whisper large-v3-turbo. Tiny, base and small can
also be selected by their catalog IDs. Large-v3 has pinned conversion metadata,
but its decoder exceeds OneModel 1.0.0's 2 GiB file limit and is refused with a
capability error; use turbo instead. Selection never silently substitutes a model.
File input is decoded by OneAudio, including AMR and AAC/M4A. Calls accept an
explicit `CancellationToken` overload. Stream input uses
`await Speech.TranscribeAsync(stream, cancellationToken)` and does not own the
stream. CPU inference is synchronous; first-use downloads can block the file call.

Each call owns and disposes its recognition and speaker models. It makes a
bounded decoding pass for whole-recording speaker clustering, then a VAD-based
recognition pass. PCM is not retained for the whole recording; the returned text,
words and turns necessarily grow with the recording. Words carry confidence and
a timing method. Whisper uses DTW word alignment; failed timing remains null.
Words outside speaker turns remain unassigned rather than receiving guessed IDs.

## First use and offline deployment

The catalog holds pinned upstream revisions, byte sizes and SHA-256 digests.
Missing tier-1 files download on first use. Until release mirror assets are
available, the library converts verified publisher weights locally, then checks
every runtime graph against its catalog size and digest before installation.
No publisher code executes during conversion. Conversion uses additional temporary
disk space and may take appreciable time for large checkpoints. Runtime files are
installed atomically per file under a cross-process cache lock. Existing verified
files require no network requests. Corrupt cache files are removed and reported;
retry the pull rather than using corrupt data.

Pre-fetch the recognizer and the speaker embedder to avoid network access in the
first transcription:

```text
onespeech models pull openai/whisper-large-v3-turbo
onespeech models pull iic/speech_campplus_sv_en_voxceleb_16k
```

Silero VAD is embedded in the core package. `ONESPEECH_MODEL_CACHE` overrides the
platform cache location. For a host-owned store, assign `ModelStore.Default` once
before starting requests, or use a separate store at the low level:

```csharp
using OneSpeech.Catalog;

ModelStore.Default = new ModelStore(cacheDirectory: cache, source: myModelSource);
```

A `ModelSource` supplies pinned publisher files; size and SHA-256 checks still
apply, and converted runtime files are checked independently. No embeddings or
enrolment audio are persisted by the library.

## Speaker features and consent

The tier-1 diarizer uses Silero VAD, CAM++ English embeddings and spectral
clustering with an AHC numerical fallback. **It does not detect overlapping
speech.** Speaker count estimation and short/changed voices can be wrong;
speaker labels are local to one recording, not verified personal identities.
An overlap-aware tier-2 upgrade is not yet available.

Voice embeddings can be biometric data. Obtain any required informed consent,
choose lawful retention and deletion practices, and assess applicable local law
before enrolling or identifying people. Illinois BIPA and GDPR Article 9 may
apply, among other laws. The library's in-memory operation is not a determination
of legal compliance; this note is not legal advice.
