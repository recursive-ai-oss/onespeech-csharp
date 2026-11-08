#!/usr/bin/env bash
set -euo pipefail
# Resumable CPU reference shards. Each queue reservation is bounded independently.
# Arguments: python publisher-model corpus-root output-root begin count
PYTHON=$1 MODEL=$2 CORPUS=$3 OUTPUT=$4 BEGIN=$5 COUNT=$6
DRIVER=$(cd "$(dirname "$0")" && pwd)/nemotron_batch.py
for chunk in 80 560; do
  for split in clean other; do
    for ((offset=BEGIN; offset<BEGIN+COUNT; offset+=25)); do
      count=$((BEGIN+COUNT-offset)); ((count>25)) && count=25
      success=0
      for attempt in 1 2 3; do
        set +e
        wq run --class heavy --cores 2 --max 25m --label "onespeech Nemotron reference $split $chunk $offset" -- "$PYTHON" "$DRIVER" "$MODEL" "$CORPUS/test-$split" "$OUTPUT/hf-$split-$chunk" "$chunk" "$offset" "$count"
        code=$?
        set -e
        if ((code==0)); then success=1; break; fi
        if ((code!=124)); then
          test ! -f "$OUTPUT/hf-$split-$chunk/error.json" || cat "$OUTPUT/hf-$split-$chunk/error.json"
          exit "$code"
        fi
      done
      ((success==1)) || exit 124
      echo "Completed reference $split ${chunk}ms [$offset,$((offset+count)))"
    done
  done
done
