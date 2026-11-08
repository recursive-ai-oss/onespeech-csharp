"""Nemotron model-card public API oracle; never inspects installed modules."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path


def main():
    import torch
    import numpy as np
    import soundfile as sf
    import nemo
    from nemo.collections.asr.models import SortformerEncLabelModel
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_path, audio_path, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    model = SortformerEncLabelModel.restore_from(str(model_path), map_location="cpu", strict=False)
    model = model.float().eval()
    model.sortformer_modules.spkcache_len = 264
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.spkcache_update_period = 300
    model._check_streaming_parameters()
    pcm, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if rate != 16000:
        raise ValueError("Expected prepared mono 16kHz input")
    if len(sys.argv) > 4:
        pcm = pcm[:int(float(sys.argv[4]) * rate)]
    # Use the card's tensor-output option; file output resumes per recording.
    with torch.inference_mode():
        segments, probs = model.diarize(audio=[pcm], batch_size=1, sample_rate=rate, include_tensor_outputs=True)
    tensor = probs[0].detach().cpu().float().numpy()
    tensor.astype("<f4").tofile(out / (audio_path.stem + ".probs"))
    (out / (audio_path.stem + ".json")).write_text(json.dumps({"nemo": nemo.__version__, "torch": torch.__version__, "precision": "fp32", "shape": list(tensor.shape), "segments": segments[0], "samples": len(pcm), "rate": rate}))
    return {"file": audio_path.name, "shape": list(tensor.shape), "segments": len(segments[0]), "nemo": nemo.__version__, "torch": torch.__version__}


if __name__ == "__main__":
    import contextlib
    try:
        # Some oracle loggers print internal tracebacks before raising. Discard their
        # console output; expose only our own bounded status, never module frames.
        with open(os.devnull, "w") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = main()
        print(json.dumps(result), flush=True)
    except Exception as error:
        result = {"error": type(error).__name__}
        if "self_attention_model='rope' is not supported" in str(error):
            result["reason"] = "Released NeMo rejects the checkpoint configuration: self_attention_model='rope' is not supported."
        print(json.dumps(result), file=sys.stderr, flush=True)
        sys.exit(1)
