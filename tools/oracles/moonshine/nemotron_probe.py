"""Small released-Transformers probes through documented processor/model APIs only.
Uses the dependency pins in the adjacent moonshine oracle uv project.
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path


def main():
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor, AutoModelForRNNT
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_dir, corpus, output = map(Path, sys.argv[1:4])
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).eval()
    for path in sorted(corpus.rglob("*.flac"))[:5]:
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000
        inputs = processor(pcm, sampling_rate=rate, language="en-US", return_tensors="pt")
        with torch.inference_mode():
            forward = model(**inputs, decoder_input_ids=torch.tensor([[13087]]), output_hidden_states=True)
            generated = model.generate(**inputs)
        inputs.input_features.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".features"))
        hidden = forward.hidden_states or ()
        for index, state in enumerate(hidden):
            state.cpu().numpy().astype("<f4").tofile(output / (path.stem + f".hidden{index}"))
        if forward.last_hidden_state is not None:
            forward.last_hidden_state.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".last"))
        tokens = generated.sequences if hasattr(generated, "sequences") else generated
        result = {"id": path.stem, "input_shape": list(inputs.input_features.shape),
                  "mask": inputs.attention_mask.tolist(), "hidden_shapes": [list(x.shape) for x in hidden],
                  "last_shape": list(forward.last_hidden_state.shape) if forward.last_hidden_state is not None else None,
                  "tokens": tokens.tolist(), "text": processor.batch_decode(tokens, skip_special_tokens=True)}
        (output / (path.stem + ".json")).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    # Framework output is discarded at the file-descriptor level, including warnings from native code.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            main()
        except Exception as error:
            out = Path(sys.argv[3])
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.json").write_text(json.dumps({"exception": type(error).__name__, "message": str(error)}))
            sys.exit(1)
