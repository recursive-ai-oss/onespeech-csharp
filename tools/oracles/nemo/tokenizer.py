"""Black-box decoded-text fixtures from the released model's public tokenizer."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path


def main():
    import torch
    from nemo.collections.asr.models import ASRModel
    torch.set_num_threads(2)
    model = ASRModel.restore_from(sys.argv[1], map_location=torch.device("cpu")).eval()
    cases = [[]] + [[i] for i in range(8192)]
    cases += [[343, 554, 651, 869, 5245, 1180, 7877], [0, 343, 0], [1, 2, 3, 4], [343, 343], [7877, 7883]]
    rows = [{"ids": ids, "text": model.tokenizer.ids_to_text(ids)} for ids in cases]
    Path(sys.argv[2]).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print("Decoded cases:", len(rows))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
