#!/usr/bin/env bash
# Qwen3-ForcedAligner vs the released toolkit on the first 200 sorted test-clean IDs, in resumable queue shards.
# run-align-rows.sh <state> <aligner-dir> <librispeech-root> <convert.dll> <oracle-python> [cores]
# Writes <state>/rows/align-ours-clean and <state>/rows/align-ref-clean (score with score.py timing).
set -euo pipefail
STATE=$1 MODEL=$2 ROOT=$3 TOOL=$4 PYTHON=$5 CORES=${6:-4}
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$STATE/rows" "$STATE/logs"
for ((begin=0; begin<200; begin+=50)); do
  wq run --class heavy --max 25m --cores "$CORES" --label "onespeech align ours $begin" -- env PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 TMPDIR="${TMPDIR:-$HOME}" DOTNET_PROCESSOR_COUNT="$CORES" \
    dotnet "$TOOL" qwen-align-batch "$MODEL" "$ROOT" test-clean "$STATE/rows/align-ours-clean" "$begin" 50 >> "$STATE/logs/align-ours-$begin.log" 2>&1
  wq run --class heavy --max 25m --cores "$CORES" --label "onespeech align reference $begin" -- env PYTHONWARNINGS=ignore TMPDIR="${TMPDIR:-$HOME}" OMP_NUM_THREADS="$CORES" MKL_NUM_THREADS="$CORES" \
    "$PYTHON" "$HERE/run.py" --model "$MODEL" --root "$ROOT" --split test-clean --output "$STATE/rows/align-ref-clean" --precision fp32 --mode align --begin "$begin" --count 50 >> "$STATE/logs/align-ref-$begin.log" 2>&1
done
