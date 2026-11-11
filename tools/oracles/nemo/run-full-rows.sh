#!/usr/bin/env bash
# Full-split Parakeet rows in resumable queue shards (both sides read the original FLAC files, sorted by ID).
# run-full-rows.sh <ours|reference> <convert.dll|checkpoint.nemo> <ids-dir> <librispeech-root> <output> <python> <lane> <lanes> <batch> <cores>
set -euo pipefail
SIDE=$1 MODEL=$2 IDS=$3 CORPUS=$4 OUTPUT=$5 PYTHON=$6 LANE=${7:-0} LANES=${8:-1} BATCH=${9:-50} CORES=${10:-4}
DIRECTORY=${PARAKEET_DIRECTORY:-}
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  total=$(wc -l < "$IDS/$split.ids")
  for ((begin=LANE*BATCH; begin<total; begin+=LANES*BATCH)); do
    count=$((total-begin)); ((count>BATCH)) && count=$BATCH
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    saved=1
    while IFS= read -r id; do [[ -f "$OUTPUT/$split/$SIDE/$id.json" ]] || { saved=0; break; }; done < <(sed -n "$((begin+1)),$((begin+count))p" "$IDS/$split.ids")
    if ((saved)); then touch "$log.done"; continue; fi
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$MODEL" parakeet-batch "$DIRECTORY" "$CORPUS/$split" "$OUTPUT/$split/$SIDE" "$begin" "$count")
    else
      command=("$PYTHON" tools/oracles/nemo/recognize.py "$MODEL" "$CORPUS/$split" "$OUTPUT/$split/$SIDE" "$begin" "$count")
    fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 25m --cores "$CORES" --label "onespeech parakeet $split $SIDE $begin" -- env PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 TMPDIR="${TMPDIR:-$HOME}" OMP_NUM_THREADS="$CORES" MKL_NUM_THREADS="$CORES" OPENBLAS_NUM_THREADS="$CORES" NUMBA_NUM_THREADS="$CORES" DOTNET_PROCESSOR_COUNT="$CORES" "${command[@]}" >> "$log" 2>&1
      code=$?
      set -e
      if ((code==0)); then complete=1; break; fi
      ((code==124)) || exit "$code"
    done
    ((complete==1)) || exit 124
    touch "$log.done"
    echo "parakeet $split $SIDE $begin: complete"
  done
done
