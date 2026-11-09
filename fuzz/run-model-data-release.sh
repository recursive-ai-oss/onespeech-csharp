#!/usr/bin/env bash
# Model-supplied byte-BPE files, bounded configurations and converter provenance.
set -euo pipefail
ulimit -c 0
work=${1:?absolute scratch directory required}
[[ "$work" = /* ]] || exit 2
mkdir -p "$work/logs" "$work/fuzz"
for target in whisper-tokenizer qwen-tokenizer whisper-config qwen-config provenance nemotron-processor whisper-generation; do
  log="$work/logs/fuzz-$target.log"
  status="$work/logs/fuzz-$target.exit"
  if [[ -f "$status" && "$(cat "$status")" = 0 ]] && awk '/^Done [0-9]+ runs in [0-9]+ second/ && $5 >= 600 { complete=1 } END { exit !complete }' "$log"; then
    echo "$target: retained completed run"; continue
  fi
  [[ ! -f "$log" ]] || { echo "Inspect previous $target log before retrying" >&2; exit 1; }
  if wq run --class heavy --max 15m --cores 4 --label "onespeech release fuzz $target" -- env ONESPEECH_FUZZ_MODEL_DATA="$target" bash fuzz/run-moonshine-smoke.sh "$work/fuzz/$target" 600 > "$log" 2>&1; then
    if ! awk '/^Done [0-9]+ runs in [0-9]+ second/ && $5 >= 600 { complete=1 } END { exit !complete }' "$log"; then echo 1 > "$status"; exit 1; fi
    echo 0 > "$status"
  else
    code=$?; echo "$code" > "$status"; echo "$target failed; inspect $log" >&2; exit "$code"
  fi
  echo "$target: complete"; grep -E 'DONE|stat::|wq: job' "$log" | tail -n 10
done
