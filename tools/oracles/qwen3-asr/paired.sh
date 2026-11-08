#!/usr/bin/env bash
# One fixed, resumable shard. Queue/resource limits are supplied by the caller.
set -euo pipefail
STATE=$1 MODEL=$2 PRECISION=$3 SPLIT=$4 BEGIN=$5 COUNT=$6
ROOT=$7
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
case "$MODEL" in 0.6b|1.7b) ;; *) exit 2;; esac
case "$PRECISION" in fp32) COMMAND=qwen-batch;; bf16) COMMAND=qwen-batch-bf16;; *) exit 2;; esac
case "$SPLIT" in test-clean|test-other) ;; *) exit 2;; esac
[ -f "$STATE/manifests/READY" ] || { echo 'Canonical artifacts are not ready.' >&2; exit 2; }
NAME="$MODEL-$PRECISION-${SPLIT#test-}"
mkdir -p "$STATE/logs"
export UV_PROJECT_ENVIRONMENT="$STATE/asr-venv" PYTHONWARNINGS=ignore
printf 'Native shard %s %s %s\n' "$NAME" "$BEGIN" "$COUNT"
dotnet "$STATE/runner-canonical/OneSpeech.Convert.dll" "$COMMAND" "$STATE/models/asr-$MODEL" "$ROOT" "$SPLIT" "$STATE/rows/ours-$NAME" "$BEGIN" "$COUNT" > "$STATE/logs/ours-$NAME-$BEGIN.log" 2>&1
printf 'Reference shard %s %s %s\n' "$NAME" "$BEGIN" "$COUNT"
uv run --frozen --project "$HERE" python "$HERE/run.py" --model "$STATE/models/asr-$MODEL" --root "$ROOT" --split "$SPLIT" --output "$STATE/rows/ref-$NAME" --precision "$PRECISION" --begin "$BEGIN" --count "$COUNT" > "$STATE/logs/ref-$NAME-$BEGIN.log" 2>&1
