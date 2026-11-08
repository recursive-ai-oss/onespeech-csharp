# Fetch and install scripts

Scripts take a caller-selected cache outside the checkout. They never upload data or credentials.

- `fetch-librispeech.sh TARGET`: pinned test-clean/test-other archives, CC BY 4.0; verify before extraction.
- `fetch-earnings22.sh TARGET`: media and original NLP transcripts at a pinned publisher commit, CC BY-SA 4.0, evaluation only. Existing files are verified, not silently replaced.
- `install-oracles.sh TARGET`: three isolated, locked CPU reference environments. Requires `uv` and Python 3.13 (uv can fetch Python); installs no system packages.

Requires Bash, curl, sha256sum, sha1sum, tar and standard Unix utilities. URLs, terms and content hashes are kept beside fetched data. The committed `pins/` files contain hashes only, not dataset content. Oracle lockfiles are in `eval/drivers/`. See [evaluation](../../docs/eval.md) for formats, settings and commands.

## Diarization data

- `fetch-ami-diarization.sh DATA_ROOT`: verifies the pinned manual-annotation v1.6.2 archive and four original Mix-Headset WAVs. AMI Full-corpus-ASR development slice ES2011a/ES2011b, holdout test slice ES2004a/ES2004b. CC BY 4.0, attribution recorded beside data.
- `fetch-alimeeting-eval.sh DATA_ROOT`: verifies the publisher's `Eval_Ali.tar.gz` (all 8 Eval sessions in the source archive). CC BY-SA 4.0, evaluation only. No corpus/derived audio or references belong in this repository.
- `prepare-diarization.sh ami|alimeeting DATA_ROOT`: fetches/verifies then runs the C# `tools/OneSpeech.DiarizationData` tool. Requires the .NET 10 SDK and caller-selected external build artifacts. Preparation overwrites only the tool's named derived outputs in the supplied data root. Do not point this at an unrelated directory.

Preparation is deterministic (`diarization-data/1`). AMI references use timed word elements, discarding untimed/non-word vocal events and zero-duration punctuation, unioning overlapping/touching words per speaker without bridging positive gaps (Landini et al., §4.2). The original headset mix's first/only channel is converted to 16 kHz PCM16 by OneAudio through AudioFileSource, without gain normalization. If resampling is required it uses the same bounded family resampler as library audio input.

AliMeeting preparation selects far-field recordings **R8001_M8004_MS801** and **R8003_M8001_MS801** and corresponding **R8001_M8004.TextGrid** / **R8003_M8001.TextGrid** from `Eval_Ali/Eval_Ali_far/`. Only regular selected archive entries are copied, with fixed flat destination names; archive links and paths are never followed. Use array **channel 1**, not the headset recordings or an unaligned array mean. Nonempty IntervalTier intervals define reference speech, with original speaker names and all positive gaps retained. This Eval slice must never be used for threshold fitting.

Each root receives `LICENSE-NOTE`, `sources.tsv` (publisher URL, licence, pinned source SHA-256), `prepared/*.wav`, `prepared/*.rttm`, and `manifest.jsonl` (exact split, meeting ID, duration, reference counts, derived audio/reference SHA-256, preparation policy). All corpus files and derived references stay outside git. Source archives are hash-pinned in `tools/scripts/pins/`, not trusted merely because a file already exists. Prepared data is not an embedding-model quality result; real-embedder AMI-dev fitting and holdout DER/JER are separate steps. See [diarization](../../docs/diarization.md) for parameter defaults, limits and the absence of overlap detection.
