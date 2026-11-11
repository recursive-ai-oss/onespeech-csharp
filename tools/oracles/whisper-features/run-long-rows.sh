#!/usr/bin/env bash
# Long-form rows, one recording per queue job; finished recordings are skipped. The native side stores each recognized chunk,
# so a job stopped at its time limit is simply resubmitted and continues.
# run-long-rows.sh <ours|reference> <integration.dll> <manifest.jsonl> <ids> <pcm-dir> <output> <python> <oracle-weights> <threads> <model-dir> <zero|default>
set -euo pipefail
SIDE=$1 TOOL=$2 MANIFEST=$3 IDS=$4 INPUT=$5 OUTPUT=$6 PYTHON=$7 WEIGHTS=$8 THREADS=$9 MODEL=${10} MODE=${11:-zero}
mkdir -p "$OUTPUT/logs"
index=0
while IFS= read -r id; do
  for attempt in 1 2 3 4 5 6; do
    [[ -f "$OUTPUT/$id.json" ]] && break
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" long-text "$MANIFEST" "$OUTPUT" "$MODEL" "$MODE" "$THREADS" "$index" 1)
    else
      # The oracle's documented transcribe(); thresholds and previous-text conditioning stay at its defaults.
      flags=(--seed 0); [[ "$MODE" = zero ]] && flags+=(--temperature-zero)
      command=("$PYTHON" tools/oracles/whisper-features/transcribe.py turbo "$INPUT" "$IDS" "$OUTPUT" --weights "$WEIGHTS" --begin "$index" --count 1 "${flags[@]}" --threads "$THREADS")
    fi
    set +e
    wq run --class heavy --max 30m --cores "$THREADS" --label "onespeech long-form $SIDE $id" -- env PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 TMPDIR="${TMPDIR:-$HOME}" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS" DOTNET_PROCESSOR_COUNT="$THREADS" "${command[@]}" >> "$OUTPUT/logs/$id.log" 2>&1
    code=$?
    set -e
    echo "$SIDE $id attempt $attempt exit $code"
  done
  index=$((index+1))
done < "$IDS"
