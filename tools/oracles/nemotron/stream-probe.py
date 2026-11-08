"""Public streaming-forward probe of the offline-style cache configuration."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings
warnings.filterwarnings("ignore")
import sys
import json
import contextlib
from pathlib import Path


def main():
    import numpy as np
    import torch
    import soundfile as sf
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor, Nemotron3DiarizationSpeakerCache
    torch.set_num_threads(4)
    root, audio, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(root, local_files_only=True, dtype=torch.float32, attn_implementation=os.environ.get("ONESPEECH_ORACLE_ATTENTION", "sdpa")).cpu().eval()
    pcm, rate = sf.read(audio, frames=90 * 16000, dtype="float32", always_2d=True)
    inputs = processor(pcm.mean(axis=1), sampling_rate=rate, return_tensors="pt")
    cache = Nemotron3DiarizationSpeakerCache(model.config.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    summaries = []
    with torch.inference_mode():
        for step in range(3):
            features = inputs.input_features[:, step * 2720:step * 2720 + 3040]
            result = model(input_features=features, speaker_cache=cache, num_lookahead_frames=40)
            result.logits.sigmoid().numpy().astype("<f4").tofile(out / ("prob" + str(step) + ".f32"))
            cache = result.speaker_cache
            stored = cache.get_embeds(torch.empty((1, 0, 512)))
            stored.numpy().astype("<f4").tofile(out / ("cache" + str(step) + ".f32"))
            features.numpy().astype("<f4").tofile(out / ("features" + str(step) + ".f32"))
            summaries.append({"step": step, "output": list(result.logits.shape), "cache": list(stored.shape)})
    return summaries


if __name__ == "__main__":
    try:
        with open(os.devnull, "w") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = main()
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
