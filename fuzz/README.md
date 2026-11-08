# Transcript JSON fuzz smoke

`OneSpeech.Fuzz` catches only the documented invalid-input exception. It uses SharpFuzz 2.2.0 and the pinned MIT-licensed libfuzzer-dotnet release v2025.05.02.0904. The runner verifies the release binary's SHA-256 and instruments only a disposable published copy of OneSpeech, never its dependencies or the build outputs used for packaging.

Run `fuzz/run-smoke.sh /absolute/scratch/fuzz 120` in an isolated CPU job. One process runs for at most 300 seconds. Seeds are synthetic JSON, not speech or biometric data. The smoke mutates inputs up to 64 KiB; dedicated tests cover the full reader size and count bounds. Findings and the grown corpus stay outside the checkout.
