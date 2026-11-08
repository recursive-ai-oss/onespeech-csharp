#!/usr/bin/env bash
# Full offline comparison, resumable at the per-utterance outputs of the existing drivers.
set -euo pipefail
TOOL=${1:?converter assembly}
PYTHON=${2:?locked Transformers oracle Python}
MODEL=${3:?pinned model directory}
CORPUS=${4:?LibriSpeech root}
OUTPUT=${5:?absolute result directory}
SHARD=${6:-500}
[[ "$SHARD" =~ ^[1-9][0-9]*$ && "$SHARD" -le 500 ]] || exit 2
mkdir -p "$OUTPUT/logs"
for split in test-clean test-other; do
  total=$(find "$CORPUS/$split" -name '*.flac' -type f | wc -l)
  for ((begin=0; begin<total; begin+=SHARD)); do
    count=$((total-begin)); ((count>SHARD)) && count=$SHARD
    for side in ours reference; do
      log="$OUTPUT/logs/$split-$side-$begin-$count.log"
      [[ ! -f "$log.done" ]] || continue
      if [[ "$side" = ours ]]; then
        command=(dotnet "$TOOL" moonshine-batch "$MODEL" "$CORPUS/$split" "$OUTPUT/$split/$side" "$begin" "$count")
      else
        command=("$PYTHON" tools/oracles/moonshine/batch.py "$MODEL" "$CORPUS/$split" "$OUTPUT/$split/$side" --begin "$begin" --count "$count")
      fi
      # A timeout preserves completed files. Another invocation resumes the same shard.
      wq run --class heavy --cores 4 --max 25m --label "onespeech offline Moonshine $split $side $begin" -- "${command[@]}" >> "$log" 2>&1
      touch "$log.done"
      echo "$split $side $begin: complete"
    done
  done
  dotnet "$TOOL" moonshine-export-slice "$CORPUS/$split" "$OUTPUT/$split/ours" "$OUTPUT/$split/reference" "$OUTPUT/$split" "$total"
done
