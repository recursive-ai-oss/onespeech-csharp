#!/usr/bin/env bash
# Scores Parakeet full-split rows and checks token and frame identity: score-parakeet.sh <eval.dll> <ids-file> <truth-map.json> <parakeet-dir> <split> <out-prefix>
set -euo pipefail
EVAL=$1 IDS=$2 MAP=$3 DIR=$4 SPLIT=$5 OUT=$6
bash "$(dirname "$0")/score-paired.sh" "$EVAL" "$IDS" "$MAP" "$DIR/$SPLIT/ours" "$DIR/$SPLIT/reference" "$OUT"
python3 - "$IDS" "$DIR/$SPLIT" <<'E'
import json, sys
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
tok = frm = total = 0
first = []
for i in ids:
    a = json.load(open(f"{sys.argv[2]}/ours/{i}.json")); b = json.load(open(f"{sys.argv[2]}/reference/{i}.json"))
    same = a["tokens"] == b["tokens"]
    tok += same; total += len(b["tokens"])
    frm += same and a["frames"] == b["timestamp"]["timestep"]
    if not same and len(first) < 5: first.append(i)
print(f"token-identical {tok}/{len(ids)}; frame-identical {frm}/{len(ids)}; reference tokens {total}; first differing {first}")
E
