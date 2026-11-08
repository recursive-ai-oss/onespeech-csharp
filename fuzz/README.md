# Transcript JSON fuzz smoke

`OneSpeech.Fuzz` catches only the documented invalid-input exception. It uses SharpFuzz 2.2.0 and the pinned MIT-licensed libfuzzer-dotnet release v2025.05.02.0904. The runner verifies the release binary's SHA-256 and instruments only a disposable published copy of OneSpeech, never its dependencies or the build outputs used for packaging.

Run `fuzz/run-smoke.sh /absolute/scratch/fuzz 120` in an isolated CPU job. One process runs for at most 600 seconds. Use 600 seconds for a release run. Seeds are synthetic JSON, not speech or biometric data. The smoke mutates inputs up to 64 KiB; dedicated tests cover the full reader size and count bounds. Findings and the grown corpus stay outside the checkout.

For the model catalog reader, set `ONESPEECH_FUZZ_CATALOG=1` before calling
`fuzz/run-smoke.sh`. This selects the catalog target and seeds the corpus with
the embedded catalog plus an empty schema-1 envelope. Use a separate work directory
from the transcript target; all other smoke bounds are unchanged.

## Tensor checkpoint inspection

`fuzz/run-checkpoint-smoke.sh /absolute/scratch/checkpoint-zip 120` mutates the complete ZIP
container. With `ONESPEECH_CHECKPOINT_PICKLE=1` and a separate scratch directory, it mutates
the pickle index inside a generated ZIP with two constant storage values. This reaches inert
tensor parsing without first having to repair a mutated ZIP directory. Both targets use the
same bounded reader, instrument only the core assembly containing the checkpoint reader, and accept only `InvalidDataException`
and ordinary I/O errors. Seed checkpoints are constructed by our harness, without audio, Python
execution or upstream fixtures. The normal executable creates seeds before instrumentation;
only libFuzzer invokes the instrumented reader. Use one process at a time.

## Release runs

All runners accept a duration up to 600 seconds (the SentencePiece runner takes
its duration as the third argument, after the vocabulary path). Run
`bash fuzz/run-release.sh /absolute/scratch /absolute/pinned/vocabulary.model`
from the repository root with .NET 10 and `wq` on PATH. Supply additional target
names to select a subset. Every target gets its own 15-minute, four-core queue
job and 600 seconds of mutation. Logs, exit codes, corpora and findings remain
under the supplied scratch directory. A completed clean log is skipped on resume;
a failed or interrupted log must be inspected before retrying.

The `wav` target focuses on RIFF/WAVE, including G.711. The `audio` target removes
the RIFF prefilter and exercises the general encoded-audio entry point. Both use
the same synthetic G.711 seeds. Only OneSpeech is instrumented, not its audio or
model-runtime dependencies. RTTM is an evaluation-tool reader, not a core reader,
and is included as a separate target.
