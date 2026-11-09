#!/usr/bin/env bash
# Full int8 Whisper comparison with matched greedy, no-word-timestamp settings.
set -euo pipefail
TOOL=$1 MODEL=$2 CORPUS=$3 OUTPUT=$4 PYTHON=$5 SIDE=$6
LANE=${7:-0} LANES=${8:-1} THREADS=${9:-8}
[[ "$SIDE" = ours || "$SIDE" = reference ]] || exit 2
batch=${10:-50}; if [[ "$SIDE" = reference && $# -lt 10 ]]; then batch=500; fi
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  mapfile -t ids < <(find "$CORPUS/$split" -name '*.flac' -type f -printf '%f\n' | sort)
  total=${#ids[@]}
  for ((begin=LANE*batch; begin<total; begin+=LANES*batch)); do
    count=$((total-begin)); ((count>batch)) && count=$batch
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    saved=1
    for id in "${ids[@]:begin:count}"; do [[ -f "$OUTPUT/$split/$SIDE/${id%.flac}.json" ]] || { saved=0; break; }; done
    if ((saved)); then touch "$log.done"; continue; fi
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" int8-whisper-batch-threads "$MODEL" "$CORPUS/$split" "$OUTPUT/$split/$SIDE" "$begin" "$count" "$THREADS")
    else
      command=("$PYTHON" tools/oracles/int8/run.py batch "$MODEL" "$OUTPUT/$split/$SIDE" --split "$CORPUS/$split" --begin "$begin" --count "$count")
    fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 25m --cores "$THREADS" --label "onespeech release int8 $split $SIDE $begin" -- env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" DOTNET_PROCESSOR_COUNT="$THREADS" "${command[@]}" >> "$log" 2>&1
      code=$?
      set -e
      if ((code==0)); then complete=1; break; fi
      ((code==124)) || exit "$code"
    done
    ((complete==1)) || exit 124
    touch "$log.done"; echo "$MODEL $split $SIDE $begin: complete"
  done
done
