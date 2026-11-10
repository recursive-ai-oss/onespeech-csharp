#!/usr/bin/env bash
# Timing revalidation shards on current code, scored later against the existing MFA truth.
# run-timing-rows.sh <state> <timing-models> <model-cache> <timing-floor.dll> <integration.dll> <method: ctc|dtw|turbo> [cores]
# <state>/<split> holds manifest.jsonl, corpus and truth (shared, read-only); outputs go to <state>/<split>/<method>.
set -euo pipefail
STATE=$1 MODELS=$2 CACHE=$3 FLOOR=$4 INTEGRATION=$5 METHOD=$6 CORES=${7:-4} BATCH=${8:-10}
for split in ${SPLITS:-test-clean test-other}; do
  total=200; [[ $split == test-other ]] && total=100
  for ((begin=0; begin<total; begin+=BATCH)); do
    if [[ "$METHOD" = turbo ]]; then
      command=(dotnet "$INTEGRATION" comparison "$STATE/$split/manifest.jsonl" "$STATE/$split/turbo" openai/whisper-large-v3-turbo "$begin" "$BATCH")
    else
      command=(dotnet "$FLOOR" native "$METHOD" "$STATE/$split" "$MODELS" "$begin" "$BATCH")
    fi
    wq run --class heavy --max 30m --cores "$CORES" --label "onespeech timing $METHOD $split $begin" -- env PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 ONESPEECH_MODEL_CACHE="$CACHE" TMPDIR="${TMPDIR:-$HOME}" DOTNET_PROCESSOR_COUNT="$CORES" "${command[@]}" || echo "shard $METHOD $split $begin exit $? (resubmit to continue)"
  done
done
