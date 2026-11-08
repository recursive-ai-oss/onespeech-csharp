#!/usr/bin/env bash
# Indicative comparisons only, run inside the CPU queue; not a publishable benchmark.
set -euo pipefail
: "${ONESPEECH_INT8_STATE:?Set the evaluation output directory}"
: "${ONESPEECH_MODELS:?Set the shared model directory}"
tool=$1
clip=$2
out=$ONESPEECH_INT8_STATE/indicative
mkdir -p "$out"
{
  uname -a
  lscpu
  dotnet --info
  printf 'OneModel=1.2.0-preview.2 OneAudio=1.7.2 GPU=unused precision=fp32/mixed-int8\n'
} > "$out/environment.txt"
for model in ${ONESPEECH_INT8_MODELS:-whisper-small whisper-large-v3-turbo moonshine-streaming-tiny}; do
  for precision in fp32 int8; do
    if [[ $precision = fp32 ]]; then directory=$ONESPEECH_MODELS/$model; else directory=$ONESPEECH_INT8_STATE/$model-int8; fi
    for threads in 1 8; do
      prefix=$out/$model-$precision-$threads
      command=int8-whisper-metrics
      [[ $model != moonshine-* ]] || command=moonshine-bakeoff-metrics
      cat /proc/loadavg > "$prefix.load"
      dotnet "$tool" "$command" "$directory" "$clip" "$prefix.json" "$threads" > "$prefix.log" 2>&1 &
      child=$!
      (while kill -0 "$child" 2>/dev/null; do cat /proc/loadavg; sleep 10; done) >> "$prefix.load" &
      monitor=$!
      status=0
      wait "$child" || status=$?
      kill "$monitor" 2>/dev/null || true
      wait "$monitor" 2>/dev/null || true
      cat /proc/loadavg >> "$prefix.load"
      [[ $status = 0 ]] || exit "$status"
    done
  done
done
