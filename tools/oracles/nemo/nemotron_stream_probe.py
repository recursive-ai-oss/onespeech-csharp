"""Public NeMo cache-aware streaming API probe for a released checkpoint."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path


def main():
    import torch
    import nemo
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ASRModel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = ASRModel.restore_from(sys.argv[1], map_location=torch.device("cpu")).float().eval()
    model.change_decoding_strategy(OmegaConf.create({"strategy": "greedy", "greedy": {"max_symbols": 10}}))
    print(json.dumps({"nemo": nemo.__version__, "torch": torch.__version__}), flush=True)
    model.eval()
    try:
        offline = model.transcribe([sys.argv[2]], batch_size=1, return_hypotheses=True)
        print(json.dumps({"offline_text": offline[0].text, "tokens": offline[0].y_sequence.tolist()}), flush=True)
    except Exception as error:
        print(json.dumps({"offline_exception": type(error).__name__, "message": str(error)}), flush=True)
    for right in [0, 1, 3, 6, 13]:
        try:
            model.eval()
            model.encoder.set_default_att_context_size([56, right])
            model.encoder.setup_streaming_params()
            print(json.dumps({"right_context": right, "streaming_config": str(model.encoder.streaming_cfg)}), flush=True)
            hypotheses = model.transcribe_simulate_cache_aware_streaming([sys.argv[2]], batch_size=1, return_hypotheses=True, online_normalization=False)
            hypothesis = hypotheses[0]
            print(json.dumps({"right_context": right, "text": hypothesis.text, "tokens": hypothesis.y_sequence.tolist()}), flush=True)
        except Exception as error:
            print(json.dumps({"right_context": right, "exception": type(error).__name__, "message": str(error)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"exception": type(error).__name__, "message": str(error)}), file=sys.stderr)
        sys.exit(1)
