"""Released-NeMo checkpoint compatibility probe; public APIs only."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import sys
import json


def main():
    import torch
    import nemo
    from nemo.collections.asr.models import ASRModel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(json.dumps({"nemo": nemo.__version__, "torch": torch.__version__, "checkpoint": os.path.basename(sys.argv[1])}), flush=True)
    model = ASRModel.restore_from(sys.argv[1], map_location=torch.device("cpu")).float().eval()
    print(json.dumps({"loaded": True, "class": type(model).__name__}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Sanitized output only: never print package frames or implementation lines.
        print(json.dumps({"loaded": False, "exception": type(error).__name__, "message": str(error)}), flush=True)
        sys.exit(1)
