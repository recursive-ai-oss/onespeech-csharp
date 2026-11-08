#!/usr/bin/env bash
# Each target is an independent bounded queue job; a completed clean log is reusable.
set -euo pipefail
work=${1:?absolute scratch directory required}
seed=${2:?pinned SentencePiece vocabulary required}
shift 2
[[ "$work" = /* && -f "$seed" ]] || exit 2
mkdir -p "$work/logs" "$work/fuzz"
targets=(transcript catalog wav audio checkpoint pickle safetensors sentencepiece moonshine parakeet rttm)
if (( $# )); then targets=("$@"); fi
for target in "${targets[@]}"; do
  case "$target" in
    transcript) command=(bash fuzz/run-smoke.sh "$work/fuzz/$target" 600);;
    catalog) command=(env ONESPEECH_FUZZ_CATALOG=1 bash fuzz/run-smoke.sh "$work/fuzz/$target" 600);;
    wav) command=(bash fuzz/run-audio-smoke.sh "$work/fuzz/$target" 600);;
    audio) command=(env ONESPEECH_FUZZ_AUDIO_ANY=1 bash fuzz/run-audio-smoke.sh "$work/fuzz/$target" 600);;
    checkpoint) command=(bash fuzz/run-checkpoint-smoke.sh "$work/fuzz/$target" 600);;
    pickle) command=(env ONESPEECH_CHECKPOINT_PICKLE=1 bash fuzz/run-checkpoint-smoke.sh "$work/fuzz/$target" 600);;
    safetensors) command=(bash fuzz/run-safetensors-smoke.sh "$work/fuzz/$target" 600);;
    sentencepiece) command=(bash fuzz/run-sentencepiece-smoke.sh "$work/fuzz/$target" "$seed" 600);;
    moonshine) command=(bash fuzz/run-moonshine-smoke.sh "$work/fuzz/$target" 600);;
    parakeet) command=(bash fuzz/run-parakeet-smoke.sh "$work/fuzz/$target" 600);;
    rttm) command=(bash fuzz/run-rttm-smoke.sh "$work/fuzz/$target" 600);;
    *) echo "Unknown target: $target" >&2; exit 2;;
  esac
  log="$work/logs/fuzz-$target.log"
  status="$work/logs/fuzz-$target.exit"
  if [[ -f "$status" && "$(cat "$status")" = 0 ]] && grep -q 'stat::number_of_executed_units:' "$log"; then
    echo "$target: retained completed run"
    continue
  fi
  # Do not overwrite a crash or an interrupted run: its log and corpus are evidence.
  [[ ! -f "$log" ]] || { echo "Inspect previous $target log before retrying" >&2; exit 1; }
  if wq run --class heavy --max 15m --cores 4 --label "onespeech release fuzz $target" -- "${command[@]}" > "$log" 2>&1; then
    echo 0 > "$status"
  else
    result=$?
    echo "$result" > "$status"
    echo "$target: failed ($result); inspect $log" >&2
    exit "$result"
  fi
  echo "$target: complete"
  grep -E 'DONE|stat::|wq: job' "$log" | tail -10
done
