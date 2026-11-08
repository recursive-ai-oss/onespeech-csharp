"""Prepare a fixed, content-pinned evaluation slice from licensed corpus data."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--references", required=True)
a = p.parse_args()
root = Path(a.root)
rows = []
for path in (root / "LibriSpeech/test-clean").rglob("*.trans.txt"):
    for line in path.read_text().splitlines():
        id_, text = line.split(" ", 1)
        audio = path.parent / (id_ + ".flac")
        rows.append((id_, text, audio))
rows.sort(key=lambda row: row[0])
if len(rows) != 2620 or len({r[0] for r in rows}) != len(rows):
    raise ValueError("Expected the complete 2620-ID test-clean set")
with open(a.manifest, "w") as manifest, open(a.references, "w") as refs:
    for id_, text, audio in rows[:100]:
        sha = hashlib.sha256(audio.read_bytes()).hexdigest()
        manifest.write(json.dumps(dict(id=id_, audio=audio.relative_to(root).as_posix(), sha256=sha), sort_keys=True) + "\n")
        refs.write(json.dumps(dict(id=id_, text=text)) + "\n")
