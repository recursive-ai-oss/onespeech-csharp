#!/usr/bin/env bash
# Run after the row's per-ID files are complete. Use the canonical evaluator project from this repository.
set -euo pipefail
state=${1:?external result directory}
evaluator=${2:-eval/OneSpeech.Eval}
mkdir -p "$state/scores"
for row in small-clean small-other turbo-clean turbo-earnings; do
  case "$row" in
    small-other) truth="$state/eval-inputs/test-other.truth.json" ;;
    turbo-earnings) truth="$state/eval-inputs/earnings.truth.jsonl" ;;
    *) truth="$state/eval-inputs/test-clean.truth.json" ;;
  esac
  dotnet run --no-build --project tools/OneSpeech.WhisperProbe -- export "$truth" "$state/$row/ours-final" "$state/$row/reference" "$state/scores/$row"
  dotnet run --no-build --project "$evaluator" -- score --ref "$state/scores/$row.truth.jsonl" --hyp "$state/scores/$row.ours.jsonl" --hyp2 "$state/scores/$row.reference.jsonl" --normalizer en --resamples 1000 --seed 1729 --out "$state/scores/$row.score.json"
done
# DTW is compared to the reference's word boundaries, not corpus-level approximate times.
dotnet run --no-build --project tools/OneSpeech.WhisperProbe -- export "$state/eval-inputs/test-clean.truth.json" "$state/small-clean/ours-timing" "$state/small-clean/reference" "$state/scores/timing"
dotnet run --no-build --project "$evaluator" -- score --ref "$state/scores/timing.reference.jsonl" --hyp "$state/scores/timing.ours.jsonl" --normalizer en --resamples 1000 --seed 1729 --out "$state/scores/timing.score.json"
