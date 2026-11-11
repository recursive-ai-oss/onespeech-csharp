#!/usr/bin/env bash
# Qwen3-ASR LibriSpeech rows over the first N sorted IDs of each split, in resumable queue shards.
# run-full-rows.sh <ours|reference> <model-dir> <librispeech-root> <output> <convert.dll|oracle-python> <lane> <lanes> <batch> <per-split-count> <cores>
set -euo pipefail
SIDE=$1 MODEL=$2 ROOT=$3 OUTPUT=$4 TOOL=$5 LANE=${6:-0} LANES=${7:-1} BATCH=${8:-50} TOTAL=${9:-1000} CORES=${10:-4}
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  for ((begin=LANE*BATCH; begin<TOTAL; begin+=LANES*BATCH)); do
    count=$((TOTAL-begin)); ((count>BATCH)) && count=$BATCH
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" qwen-batch "$MODEL" "$ROOT" "$split" "$OUTPUT/$split/ours" "$begin" "$count")
    else
      command=(env PYTHONWARNINGS=ignore "$TOOL" "$HERE/run.py" --model "$MODEL" --root "$ROOT" --split "$split" --output "$OUTPUT/$split/reference" --precision fp32 --begin "$begin" --count "$count")
    fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 30m --cores "$CORES" --label "onespeech qwen1.7b $split $SIDE $begin" -- env PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 TMPDIR="${TMPDIR:-$HOME}" OMP_NUM_THREADS="$CORES" MKL_NUM_THREADS="$CORES" DOTNET_PROCESSOR_COUNT="$CORES" "${command[@]}" >> "$log" 2>&1
      code=$?
      set -e
      if ((code==0)); then complete=1; break; fi
      ((code==124)) || exit "$code"
    done
    ((complete==1)) || exit 124
    touch "$log.done"
    echo "qwen $split $SIDE $begin: complete"
  done
done
