#!/usr/bin/env bash
# Public-role/default Whisper policy, matching the released oracle's public options.
set -euo pipefail
TOOL=$1 MODEL=$2 NAME=$3 INPUT=$4 OUTPUT=$5 PYTHON=$6 WEIGHTS=$7 SIDE=$8
LANE=${9:-0} LANES=${10:-1}
[[ "$SIDE" = ours || "$SIDE" = reference ]] || exit 2
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  total=$(wc -l < "$INPUT/$split.ids")
  for ((begin=LANE*50; begin<total; begin+=LANES*50)); do
    count=$((total-begin)); ((count>50)) && count=50
    log="$OUTPUT/logs/$split-$SIDE-$begin-$count.log"
    [[ ! -f "$log.done" ]] || continue
    if [[ "$SIDE" = ours ]]; then
      command=(dotnet "$TOOL" recognize-text "$MODEL" "$INPUT/$split" "$INPUT/$split.ids" "$OUTPUT/$split/$SIDE" "$begin" "$count" words)
    else
      command=("$PYTHON" tools/oracles/whisper-features/transcribe.py "$NAME" "$INPUT/$split" "$INPUT/$split.ids" "$OUTPUT/$split/$SIDE" --weights "$WEIGHTS" --begin "$begin" --count "$count" --words --seed 0)
    fi
    complete=0
    for attempt in 1 2 3; do
      set +e
      wq run --class heavy --max 25m --cores 8 --label "onespeech release $NAME $split $SIDE $begin" -- "${command[@]}" >> "$log" 2>&1
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
