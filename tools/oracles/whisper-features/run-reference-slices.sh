#!/usr/bin/env bash
# Per-ID files make every bounded shard resumable. Substitute full ID manifests at release evaluation.
set -euo pipefail
model=${1:?small or turbo}
split=${2:?test-clean or test-other}
state=${3:?external state directory}
weights=${4:?official weight cache}
python=${5:?oracle Python executable}
words=()
name="$model-${split#test-}"
if [[ "$model" = small && "$split" = test-clean ]]; then words=(--words); fi
total=$(wc -l < "$state/eval-inputs/$split.ids")
for ((begin=0;begin<total;begin+=20)); do
  wq run --class heavy --max 30m --cores 8 --label "onespeech u1 reference $name $begin" -- \
    "$python" tools/oracles/whisper-features/transcribe.py "$model" "$state/eval-inputs/$split" "$state/eval-inputs/$split.ids" "$state/$name/reference" \
    --weights "$weights" --begin "$begin" --count 20 "${words[@]}"
done
