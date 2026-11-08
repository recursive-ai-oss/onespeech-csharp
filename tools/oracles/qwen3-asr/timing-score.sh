#!/usr/bin/env bash
# Compare fixed-slice boundaries and include every missing prediction in the report.
set -euo pipefail
STATE=$1
shift
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export UV_PROJECT_ENVIRONMENT="$STATE/asr-venv"
uv run --frozen --project "$HERE" python "$HERE/score.py" timing "$STATE" "$@"
