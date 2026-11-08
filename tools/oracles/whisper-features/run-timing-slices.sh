#!/usr/bin/env bash
set -euo pipefail
model=${1:?model directory}
state=${2:?external results directory}
total=$(wc -l < "$state/eval-inputs/test-clean.ids")
for ((begin=0;begin<total;begin+=20)); do
  wq run --class heavy --max 30m --cores 8 --label "onespeech u1 final DTW $begin" -- \
    dotnet run --no-build --project tools/OneSpeech.WhisperProbe -- realign "$model" "$state/eval-inputs/test-clean" "$state/eval-inputs/test-clean.ids" "$state/small-clean/ours-final" "$state/small-clean/ours-timing" "$begin" 20
done
