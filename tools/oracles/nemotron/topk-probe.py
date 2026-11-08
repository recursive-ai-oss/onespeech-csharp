"""Released Torch top-k tie observations on synthetic score vectors, no model or source inspection."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")
import contextlib
import json
import sys
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for length, k in ((300, 24), (300, 48), (604, 24), (604, 48), (2408, 264), (4840, 264)):
        for kind in ("equal", "bands"):
            scores = torch.zeros(length, dtype=torch.float32)
            if kind == "bands":
                if length in (300, 604):
                    scores[264:] = .05
                else:
                    width = length // 8
                    scores.fill_(float("-inf"))
                    scores[:width - 1] = 0
                    scores[264:width - 1] = .05
                    scores[width - 1::width] = float("inf")
            indices = torch.topk(scores, k, largest=True, sorted=True).indices
            name = f"{length}-{k}-{kind}"
            scores.numpy().astype("<f4").tofile(out / (name + ".scores"))
            indices.numpy().astype("<i8").tofile(out / (name + ".ids"))
            rows.append({"name": name, "length": length, "k": k})
    (out / "manifest.json").write_text(json.dumps({"torch": torch.__version__, "rows": rows}))
    return {"torch": torch.__version__, "cases": len(rows)}


if __name__ == "__main__":
    try:
        with open(os.devnull, "w") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = main()
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
