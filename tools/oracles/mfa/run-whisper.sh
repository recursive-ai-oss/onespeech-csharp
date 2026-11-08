#!/usr/bin/env bash
set -euo pipefail
state=${1:?external state directory}
weights=${2:?original OpenAI weight cache}
python=${3:?Python from the locked whisper-features oracle environment}
splits=(test-clean test-other)
if [[ ${4:-} != '' ]]; then splits=("$4"); fi
for split in "${splits[@]}"; do
  [[ $split == test-clean || $split == test-other ]] || { echo 'Expected test-clean or test-other' >&2; exit 2; }
  total=200; [[ $split == test-other ]] && total=100
  for ((begin=0; begin<total; begin+=25)); do
    wq run --class heavy --max 30m --cores 4 --label "onespeech timing whisper oracle $split $begin" -- \
      "$python" "$(dirname "$0")/oracle_whisper.py" "$state/$split" "$weights" "$begin" 25
  done
done
