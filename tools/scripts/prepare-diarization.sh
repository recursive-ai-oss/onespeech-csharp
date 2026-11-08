#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?Usage: prepare-diarization.sh ami|alimeeting DATA_ROOT}
target=${2:?DATA_ROOT required}
repo=$(cd "$(dirname "$0")/../.." && pwd)
case "$dataset" in
  ami) bash "$repo/tools/scripts/fetch-ami-diarization.sh" "$target" ;;
  alimeeting) bash "$repo/tools/scripts/fetch-alimeeting-eval.sh" "$target" ;;
  *) echo 'Expected ami or alimeeting.' >&2; exit 2 ;;
esac
dotnet run --project "$repo/tools/OneSpeech.DiarizationData" -- "$dataset" "$target"
