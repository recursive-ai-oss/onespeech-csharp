#!/usr/bin/env bash
# Public-role/default Whisper policy, matching the released oracle's public options.
set -euo pipefail
TOOL=$1 MODEL=$2 NAME=$3 INPUT=$4 OUTPUT=$5 PYTHON=$6 WEIGHTS=$7 SIDE=$8
LANE=${9:-0} LANES=${10:-1} THREADS=${11:-8} BATCH=${12:-50} CORPUS=${13:-}
[[ "$SIDE" = ours || "$SIDE" = reference ]] || exit 2
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  total=$(wc -l < "$INPUT/$split.ids")
  for ((begin=LANE*BATCH; begin<total; begin+=LANES*BATCH)); do
    count=$((total-begin)); ((count>BATCH)) && count=$BATCH
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    saved=1
    while IFS= read -r id; do [[ -f "$OUTPUT/$split/$SIDE/$id.json" ]] || { saved=0; break; }; done < <(sed -n "$((begin+1)),$((begin+count))p" "$INPUT/$split.ids")
    if ((saved)); then touch "$log.done"; continue; fi
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" recognize-text "$MODEL" "$INPUT/$split" "$INPUT/$split.ids" "$OUTPUT/$split/$SIDE" "$begin" "$count" words default 0 "$THREADS")
    else
      command=("$PYTHON" tools/oracles/whisper-features/transcribe.py "$NAME" "$INPUT/$split" "$INPUT/$split.ids" "$OUTPUT/$split/$SIDE" --weights "$WEIGHTS" --begin "$begin" --count "$count" --words --seed 0 --threads "$THREADS")
    fi
    if [[ "$SIDE" = reference && -n "$CORPUS" ]]; then command+=(--corpus "$CORPUS/$split"); fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 25m --cores "$THREADS" --label "onespeech release $NAME $split $SIDE $begin" -- env OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" DOTNET_PROCESSOR_COUNT="$THREADS" "${command[@]}" >> "$log" 2>&1
      code=$?
      set -e
      if ((code==0)); then complete=1; break; fi
      ((code==124)) || exit "$code"
    done
    ((complete==1)) || exit 124
    touch "$log.done"
    echo "$NAME $split $SIDE $begin: complete"
  done
done

# All selected shards completed or were already present.
exit 0
