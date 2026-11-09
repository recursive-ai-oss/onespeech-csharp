"""Black-box Nemotron diarization reference using the released Transformers public API."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings
warnings.filterwarnings("ignore")
import sys
import json
import contextlib
from pathlib import Path


phase = "imports"


def main():
    global phase
    import numpy as np
    import torch
    import soundfile as sf
    import transformers
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_dir, audio_path, output = map(Path, sys.argv[1:4])
    output.mkdir(parents=True, exist_ok=True)
    phase = "processor"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    phase = "model"
    attention = os.environ.get("ONESPEECH_ORACLE_ATTENTION", "sdpa")
    model = AutoModelForAudioFrameClassification.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32, attn_implementation=attention).cpu().eval()
    pcm, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if rate != 16000:
        raise ValueError("Expected prepared 16kHz audio")
    if len(sys.argv) > 4:
        pcm = pcm[:int(float(sys.argv[4]) * rate)]
    phase = "features"
    inputs = processor(pcm, sampling_rate=rate, return_tensors="pt")
    phase = "forward"
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=(len(sys.argv) > 5))
    logits = result.logits.cpu().float()
    logits.sigmoid().numpy().astype("<f4").tofile(output / (audio_path.stem + ".probs"))
    phase = "segments"
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    metadata = {"transformers": transformers.__version__, "torch": torch.__version__, "precision": "fp32", "attention": attention, "shape": list(logits.shape), "segments": segments, "samples": len(pcm), "rate": rate}
    (output / (audio_path.stem + ".json")).write_text(json.dumps(metadata))
    with (output / (audio_path.stem + ".rttm")).open("w") as rttm:
        for segment in segments:
            rttm.write(f"SPEAKER {audio_path.stem} 1 {segment['Start']:.9f} {segment['End'] - segment['Start']:.9f} <NA> <NA> speaker_{segment['Speaker']} <NA> <NA>\n")
    if len(sys.argv) > 5:
        with torch.inference_mode():
            bare = model.model(input_features=inputs.input_features, attention_mask=inputs.attention_mask)
        bare.last_hidden_state.cpu().float().numpy().astype("<f4").tofile(output / "upsampled.f32")
        inputs.input_features.cpu().numpy().astype("<f4").tofile(output / "features.f32")
        metadata["upsampled_shape"] = list(bare.last_hidden_state.shape)
        metadata["features_shape"] = list(inputs.input_features.shape)
        metadata["hidden_shapes"] = [list(h.shape) for h in result.hidden_states]
        for i, h in enumerate(result.hidden_states):
            h.cpu().float().numpy().astype("<f4").tofile(output / ("hidden-" + str(i) + ".f32"))
        (output / "probe.json").write_text(json.dumps(metadata))
    return {"audio": audio_path.name, "shape": list(logits.shape), "transformers": transformers.__version__, "torch": torch.__version__}


if __name__ == "__main__":
    try:
        with open(os.devnull, "w") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                status = main()
        print(json.dumps(status), flush=True)
    except Exception as error:
        # No module frames, source lines, or library exception text are exposed.
        print(json.dumps({"error": type(error).__name__, "phase": phase}), file=sys.stderr, flush=True)
        sys.exit(1)
