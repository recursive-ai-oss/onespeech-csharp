# Command line and offline model packages

Speech recognition for .NET. `onespeech --help` lists commands, positional arguments,
options and exit codes. The tool accepts standalone audio and OneMedia containers.
`transcribe` and `align` emit text, schema-v1 JSON, SRT or VTT (`--format`);
`--output` selects a UTF-8 file instead of stdout. `align` reads its second argument
as a UTF-8 text file. Model and language options are passed to the library, with
unsupported requests reported as typed errors rather than ignored.

`diarize` uses the low-level clustering defaults and returns turns without speech
recognition. Its text output is start seconds, end seconds and speaker ID, separated
by tabs. JSON uses the same Transcript schema; SRT/VTT contain speaker-label cues,
not recognized words. `--model nvidia/Nemotron-3-Diarization` selects the overlap-aware
alternative. The default clustering method does not detect overlap.

`identify query.wav alice enrollment.wav bob another.wav` holds all enrollments
only until that invocation exits. `verify first.wav second.wav` prints a cosine
score, not a probability or a calibrated decision. `enroll alice enrollment.wav`
checks enrollment and discards it on exit; it does not create a database.
Speaker samples are limited to 60 seconds per file. The default identify threshold
is 0.3; `--threshold` accepts a finite cosine in [-1,1]. It is not a population
calibration. Obtain informed consent and check local law (including BIPA and GDPR
Art. 9) before processing voices.

`models list` is offline. `models pull <id>` prefetches a pinned model; gate tokens
are read from `ONESPEECH_MODEL_TOKEN`, never a command-line argument or saved file.
Exit codes are 0 (success), 1 (model/audio/I/O failure), 2 (usage), 130 (cancelled).
Errors print their typed name and message to stderr; inference never writes logs
or progress to stdout. Ctrl+C cancels through the library's cancellation points.

## Offline packages

Reference `RecursiveAI.OneSpeech.Models.CamPlusEnglish`, `.MoonshineTiny`,
`.MoonshineBase`, or `.MoonshineStreamingTiny` alongside core. Each contains only
approved, pinned tier-1 publisher files, its notices and an AOT-safe transitive
build target. No reflection or assembly registration is used. Build and publish
copy the files into `onespeech-models/<catalog SHA-256>` beside the executable.
Keep that directory, including when deploying NativeAOT or a single-file app.

The loader tries these files before its configured ModelSource, verifies their
size and SHA-256, copies them atomically into the per-user cache, then performs
normal local conversion and verifies converted hashes. Corrupt package content
fails rather than falling back to network. Tier-2 models never read package
content. A package covers only its own model, not all roles in `Speech.Transcribe`
(which may also need an aligner and diarizer). `tests/OfflinePackageSmoke` demonstrates
low-level transcription with network access explicitly refused by its ModelSource.
NuGet restore itself needs the packages and their dependencies in a local feed/cache.

Maintainers prepare and pack via `tools/models/pack-model-packages.sh`. Set
`ONESPEECH_ARTIFACTS`, `MODEL_PACKAGE_CACHE` and `MODEL_PACKAGE_FEED` to external
directories. Optionally set `MODEL_PUBLISHER_CACHE` to an existing publisher-file
cache. The C# preparer fetches missing files, validates each against the embedded
catalog, and revalidates immediately before pack. No weights are stored in git.
All packages share the repository version and metadata; each archive must remain
below 250 MB. Nothing is published by the script.

Unaffiliated with other products or companies named OneSpeech.

## Command examples

After installing the tool from a feed containing its package and dependencies:

```sh
onespeech --help
onespeech models list
onespeech models pull openai/whisper-tiny
onespeech transcribe meeting.m4a --language en --format json --output meeting.json
onespeech align short.wav words.txt --language en --format vtt --output short.vtt
onespeech diarize meeting.wav --model nvidia/Nemotron-3-Diarization
onespeech identify query.wav alice enrollment.wav --threshold 0.3
onespeech verify first.wav second.wav
```

Model IDs select roles, not filenames. An ASR model passed to `diarize` is not a speaker embedder. First use may download additional models for alignment and diarization; pre-fetch those as well for a disconnected host. Gate tokens must be injected through the environment, never added to these command lines. Enrollment and query files must be obtained with appropriate consent.
