"""Serialize saved public-pipeline turns without repeating neural inference.

Malformed nonpositive intervals are rejected unless explicitly omitted. Omission is
reported with the original endpoints; it never swaps endpoints or invents timings.
"""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("output", type=Path)
p.add_argument("recording")
p.add_argument("--omit-nonpositive", action="store_true")
a = p.parse_args()
turns = json.loads(a.input.read_text())
lines, omitted = [], []
for turn in turns:
    start, end = float(turn[0]), float(turn[1])
    speaker = str(turn[2]).replace(" ", "_")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0:
        raise ValueError("Nonfinite/negative public-pipeline endpoint")
    if end <= start:
        if not a.omit_nonpositive:
            raise ValueError("Nonpositive public-pipeline interval")
        omitted.append({"start": start, "end": end, "speaker": speaker})
        continue
    lines.append(f"SPEAKER {a.recording} 1 {start:.9f} {end-start:.9f} <NA> <NA> {speaker} <NA> <NA>\n")
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text("".join(lines))
print(json.dumps({"input": str(a.input), "output": str(a.output), "turns": len(turns), "written": len(lines), "omitted": omitted}))
