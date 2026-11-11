#!/usr/bin/env bash
# Scores timing hypotheses (ctc|dtw|turbo|oracle) of a prepared split against MFA truth and prints pooled boundary errors.
# score-timing.sh <timing-floor.dll> <eval.dll> <split-dir> <truth.jsonl> <method>...
set -euo pipefail
FLOOR=$1 EVAL=$2 SPLIT=$3 TRUTH=$4; shift 4
export PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1
for method in "$@"; do
  [[ -f "$SPLIT/$method.jsonl" ]] || dotnet "$FLOOR" export-method "$SPLIT" "$method"
  dotnet "$EVAL" score --ref "$TRUTH" --hyp "$SPLIT/$method.jsonl" --normalizer en --resamples 1000 --seed 1729 --out "$SPLIT/$method-score.json" > /dev/null
  echo "$(basename "$SPLIT") $method: $(dotnet "$FLOOR" pool "$SPLIT/$method-score.json")"
done
