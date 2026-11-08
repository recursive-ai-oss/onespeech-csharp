#!/usr/bin/env bash
set -euo pipefail
model=${1:?model directory}
split=${2:?test-clean or test-other}
state=${3:?external results directory}
name=${4:?row name}
mode=no-words
if [[ "$name" = small-clean ]]; then mode=words; fi
total=$(wc -l < "$state/eval-inputs/$split.ids")
for ((begin=0;begin<total;begin+=20)); do
  wq run --class heavy --max 30m --cores 8 --label "onespeech u1 native $name $begin" -- \
    dotnet run --no-build --project tools/OneSpeech.WhisperProbe -- recognize "$model" "$state/eval-inputs/$split" "$state/eval-inputs/$split.ids" "$state/$name/ours-final" "$begin" 20 "$mode"
done
