# Forced-alignment timing floors

These scripts measure absolute word-boundary errors against MFA truth on the
first 200 ordinal LibriSpeech test-clean IDs and first 100 test-other IDs. They
write all corpus derivatives, alignments, hypotheses and reports outside git.
LibriSpeech is CC-BY-4.0; retain its corpus attribution with the external state.

## Oracle installation

Use released micromamba 2.3.3, then `bash install-conda.sh "$MFA_ENV"` with
`MICROMAMBA` set to its executable. The Linux x64 explicit file locks every
binary package URL, version/build and MD5. It includes MFA 3.4.3, Kalpy 0.10.5,
CPU Kaldi 5.5.1172 and Python 3.12. No package is built from source. Never inspect
oracle modules. The older uv project/check.py records the unsupported wheel-only
installation route; the CLI runs in the binary environment instead.

Fetch the English MFA acoustic model and dictionary with the existing
`OneSpeech.Convert mfa-fetch` command. Both are publisher v3.0.0 assets, pinned
by SHA-256 in MfaAssets.cs and checked again by run.py. Keep ATTRIBUTION.txt with
them: Michael McAuliffe and Morgan Sonderegger (2024), Montreal Corpus Tools,
CC-BY-4.0. These truth assets are not redistributed.

## Reproduction

From the repository root, set external `STATE`, `MODELS`, `MFA_ENV`, `MFA_ASSETS`,
`TMPDIR`, `ONESPEECH_ARTIFACTS`, and the usual .NET environment. `PYTHON` below is
from the locked `../whisper-features` uv oracle project (SoundFile 0.14.0,
NumPy 2.5.4, openai-whisper 20250625, CPU Torch 2.15.0).

```sh
"$PYTHON" tools/oracles/mfa/prepare.py "$LIBRISPEECH/LibriSpeech" "$STATE"
for split in test-clean test-other; do
  wq run --class heavy --max 30m --cores 4 --label "onespeech MFA $split" -- \
    "$PYTHON" tools/oracles/mfa/run.py "$MFA_ENV" "$MFA_ASSETS" "$STATE/$split"
done
dotnet build tools/OneSpeech.TimingFloor -m:4
bash tools/oracles/mfa/run-native.sh "$STATE" "$MODELS"
bash tools/oracles/mfa/run-whisper.sh "$STATE" "$OPENAI_WEIGHTS" "$PYTHON"
for split in test-clean test-other; do
  dotnet run --no-build --project tools/OneSpeech.TimingFloor -- export "$STATE/$split"
  for method in ctc dtw oracle; do
    dotnet run --no-build --project eval/OneSpeech.Eval -- score \
      --ref "$STATE/$split/truth.jsonl" --hyp "$STATE/$split/$method.jsonl" \
      --normalizer en --resamples 1000 --seed 1729 --out "$STATE/$split/$method-score.json"
  done
done
```

Run CPU batches through the queue. Native/oracle batches resume per atomic
utterance file; MFA uses its documented temporary directory and output files.
The native cache alias points to the existing verified omniASR model files;
ModelStore still verifies the published and converted hashes. No weights are
copied or uploaded. CTC uses public `Speech.Align(path, referenceText, language:
"en", model: "facebook/omniASR-CTC-300M")`. DTW uses the low-level Whisper small
recognizer, not Speech.Transcribe, so the pipeline cannot replace DTW with CTC.
Both Whisper runs are fp32 greedy, temperature 0, English transcription,
default rejection thresholds and word timestamps enabled. Native Whisper uses
four graph threads; the public Speech.Align loader uses its one-thread default.
The oracle runner accepts an optional fourth argument selecting a single split,
so disjoint split runs can be queued independently. No speed claim is made.

## Scoring

`export` converts MFA's public JSON word tier and the native/oracle outputs to
main's onespeech-eval JSONL format. It fails if a requested file is missing.
MFA speaker adaptation is explicitly disabled; no acoustic model is fitted or
fine-tuned. Silence entries are not words. Recognition differences remain
visible: this is not a claim that Whisper forcibly aligns arbitrary supplied
text. Whisper is transcribed, then words are matched to truth using the same
`en/1` per-word normalizer and minimum-edit alignment as `TimingMetrics.Score`.

The main scorer reports per-utterance metrics. `summary.json` calls that same
scorer on each pre-matched pair, then pools its errors to report word-weighted means/medians and the
fractions within 50/100 ms (inclusive with its 1e-9 tolerance), separately for
start and end. No median-of-medians is used. `allMatched` includes every equal
word pair for that method; missing times are counted, not fabricated.
`common` includes only truth-word indices matched and timed by all three
methods, providing a literal same-word comparison. Unmatched truth words and
untimed hypothesis words are reported separately. Missing and unmatched words
are excluded from error denominators, never silently counted as correct.
These are first timing floors, not human-annotated ground truth or latency /
speed / quality-parity claims. The floor does not itself change timing policy.
