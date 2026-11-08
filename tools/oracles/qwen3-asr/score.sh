#!/usr/bin/env bash
# Pair the same first 200 sorted IDs; incomplete rows are reported, never scored.
set -euo pipefail
STATE=$1 ROOT=$2
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export UV_PROJECT_ENVIRONMENT="$STATE/asr-venv"
uv run --frozen --project "$HERE" python "$HERE/score.py" wer "$STATE" "$ROOT"
