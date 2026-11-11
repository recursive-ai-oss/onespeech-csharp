#!/usr/bin/env bash
set -euo pipefail
# Arguments: dotnet-tool publisher-converted corpus-root output-root begin count [push-samples [shard-size]]
# push-samples is the size of each PCM push in samples (default: one attention chunk); shard-size defaults to 25 utterances.
TOOL=$1 MODEL=$2 CORPUS=$3 OUTPUT=$4 BEGIN=$5 COUNT=$6 PUSH=${7:-} STEP=${8:-25}
for chunk in ${CHUNKS:-80 560}; do
  for split in ${SPLITS:-clean other}; do
    for ((offset=BEGIN; offset<BEGIN+COUNT; offset+=STEP)); do
      count=$((BEGIN+COUNT-offset)); ((count>STEP)) && count=$STEP
      success=0
      for attempt in 1 2 3; do
        set +e
        wq run --class heavy --cores 4 --max 25m --label "onespeech Nemotron native $split $chunk $offset" -- dotnet "$TOOL" nemotron-asr-live-batch "$MODEL" "$CORPUS/test-$split" "$OUTPUT/ours-$split-live-$chunk" "$chunk" "$offset" "$count" $PUSH
        code=$?
        set -e
        if ((code==0)); then success=1; break; fi
        ((code==124)) || exit "$code"
      done
      ((success==1)) || exit 124
    done
  done
done
