"""Compare public long-form output segments; evaluation diagnostics only."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("ours", type=Path)
parser.add_argument("reference", type=Path)
args = parser.parse_args()
ours = json.loads(args.ours.read_text())["transcript"]
reference = json.loads(args.reference.read_text())
left = [s for s in ours["segments"] if s["text"].strip()]
right = [s for s in reference["segments"] if s["text"].strip()]
print("segment counts", len(left), len(right))
for i, (a, b) in enumerate(zip(left, right)):
    if a["text"] != b["text"] or abs(a["start"] - b["start"]) > .001 or abs(a["end"] - b["end"]) > .001:
        print("first differing nonempty segment", i)
        for j in range(max(0, i - 2), min(i + 5, len(left), len(right))):
            print("ours", j, json.dumps({k: v for k, v in left[j].items() if k != "words"}, ensure_ascii=False))
            print("reference", j, json.dumps(right[j], ensure_ascii=False))
        break
else:
    print("all paired segment texts identical")
