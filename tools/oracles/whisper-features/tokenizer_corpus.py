"""Prepare crafted and evaluation-transcript tokenizer cases as evaluation data."""
import warnings
warnings.simplefilter("ignore")
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
values = json.loads(Path(sys.argv[2]).read_text())
for split in ["test-clean", "test-other"]:
    values.extend(json.loads((root / (split + ".truth.json")).read_text()).values())
Path(sys.argv[3]).write_text(json.dumps(values, ensure_ascii=False) + "\n")
