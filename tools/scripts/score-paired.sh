#!/usr/bin/env bash
# Scores one paired WER row with the shared metric tool (en normalizer, 1000 resamples, seed 1729).
# score-paired.sh <eval.dll> <ids-file> <id->text map.json> <ours-dir> <reference-dir> <output-prefix>
# Prints the score line and the number of utterances whose raw texts differ.
set -euo pipefail
EVAL=$1 IDS=$2 MAP=$3 OURS=$4 REFERENCE=$5 OUT=$6
export PATH="$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1
dotnet "$EVAL" collect --ids "$IDS" --map "$MAP" --out "$OUT.truth.jsonl" > /dev/null
dotnet "$EVAL" collect --ids "$IDS" --dir "$OURS" --out "$OUT.ours.jsonl" > /dev/null
dotnet "$EVAL" collect --ids "$IDS" --dir "$REFERENCE" --out "$OUT.reference.jsonl" > /dev/null
dotnet "$EVAL" score --ref "$OUT.truth.jsonl" --hyp "$OUT.ours.jsonl" --hyp2 "$OUT.reference.jsonl" --normalizer en --resamples 1000 --seed 1729 --out "$OUT.score.json"
echo "utterances $(wc -l < "$OUT.truth.jsonl"); raw text differs in $(diff "$OUT.ours.jsonl" "$OUT.reference.jsonl" | grep -c '^<' || true)"
