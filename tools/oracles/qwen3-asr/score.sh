#!/usr/bin/env bash
# Pair the same first 200 sorted IDs; no missing hypotheses are discarded.
set -euo pipefail
STATE=$1 ROOT=$2
mkdir -p "$STATE/scores"
for SPLIT in clean other; do
    find "$ROOT/test-$SPLIT" -name '*.trans.txt' -type f -print0 | sort -z | xargs -0 cat | LC_ALL=C sort | awk 'NR<=200{print}' | jq -Rc 'capture("^(?<id>[^ ]+) (?<text>.*)$")' > "$STATE/scores/truth-$SPLIT.jsonl"
    for MODEL in 0.6b 1.7b; do
        for PRECISION in fp32 bf16; do
            NAME="$MODEL-$PRECISION-$SPLIT"
            [ -d "$STATE/rows/ours-$NAME" ] && [ -d "$STATE/rows/ref-$NAME" ] || continue
            for SIDE in ours ref; do
                jq -sc '.[] | select(has("id"))' "$STATE/rows/$SIDE-$NAME/"*.json > "$STATE/scores/$SIDE-$NAME.jsonl"
            done
            [ "$(wc -l < "$STATE/scores/ours-$NAME.jsonl")" -eq 200 ] || continue
            [ "$(wc -l < "$STATE/scores/ref-$NAME.jsonl")" -eq 200 ] || continue
            dotnet "$STATE/runner-eval/OneSpeech.Eval.dll" score --ref "$STATE/scores/truth-$SPLIT.jsonl" --hyp "$STATE/scores/ours-$NAME.jsonl" --hyp2 "$STATE/scores/ref-$NAME.jsonl" --normalizer en --resamples 1000 --seed 1729 --out "$STATE/scores/$NAME.json" > "$STATE/scores/$NAME.log"
            printf '%s: ' "$NAME"; head -2 "$STATE/scores/$NAME.log" | tr '\n' ' '; printf '\n'
        done
    done
done
