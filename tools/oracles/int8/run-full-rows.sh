#!/usr/bin/env bash
# Full int8 Whisper comparison with matched greedy, no-word-timestamp settings.
set -euo pipefail
TOOL=$1 MODEL=$2 CORPUS=$3 OUTPUT=$4 PYTHON=$5 SIDE=$6
LANE=${7:-0} LANES=${8:-1}
[[ "$SIDE" = ours || "$SIDE" = reference ]] || exit 2
batch=50; [[ "$SIDE" = reference ]] && batch=500
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  total=$(find "$CORPUS/$split" -name '*.flac' -type f | wc -l)
  for ((begin=LANE*batch; begin<total; begin+=LANES*batch)); do
    count=$((total-begin)); ((count>batch)) && count=$batch
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" int8-whisper-batch "$MODEL" "$CORPUS/$split" "$OUTPUT/$split/$SIDE" "$begin" "$count")
    else
      command=("$PYTHON" tools/oracles/int8/run.py batch "$MODEL" "$OUTPUT/$split/$SIDE" --split "$CORPUS/$split" --begin "$begin" --count "$count")
    fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 25m --cores 8 --label "onespeech release int8 $split $SIDE $begin" -- "${command[@]}" >> "$log" 2>&1
      code=$?
      set -e
      if ((code==0)); then complete=1; break; fi
      ((code==124)) || exit "$code"
    done
    ((complete==1)) || exit 124
    touch "$log.done"; echo "$MODEL $split $SIDE $begin: complete"
  done
done
