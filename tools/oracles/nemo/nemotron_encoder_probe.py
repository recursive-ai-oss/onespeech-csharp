"""NeMo public encoder probes for cache-aware FastConformer; no implementation inspection."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path


def main():
    import torch
    import numpy as np
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_path, corpus, output = map(Path, sys.argv[1:4])
    output.mkdir(parents=True, exist_ok=True)
    model = ASRModel.restore_from(str(model_path), map_location=torch.device("cpu")).float().eval()
    model.encoder.set_default_att_context_size([56, 3])
    model.change_decoding_strategy(OmegaConf.create({"strategy": "greedy", "greedy": {"max_symbols": 10}}))
    for path in sorted(corpus.rglob("*.flac"))[:5]:
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000
        model.eval()
        with torch.inference_mode():
            mel, lengths = model.preprocessor(input_signal=torch.from_numpy(pcm)[None], length=torch.tensor([len(pcm)]))
            encoded, encoded_lengths = model.encoder(audio_signal=mel, length=lengths)
        pcm.astype("<f4").tofile(output / (path.stem + ".pcm"))
        mel.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".mel"))
        encoded.cpu().numpy().astype("<f4").tofile(output / (path.stem + ".encoded"))
        print(json.dumps({"id": path.stem, "samples": len(pcm), "mel_shape": list(mel.shape), "mel_length": int(lengths[0]), "encoder_shape": list(encoded.shape), "encoder_length": int(encoded_lengths[0])}), flush=True)
    # Manifest is documented input data; the prompt field comes from the published model config.
    path = sorted(corpus.rglob("*.flac"))[1]
    manifest = output / "prompt-manifest.json"
    pcm, rate = sf.read(path, dtype="float32")
    manifest.write_text(json.dumps({"audio_filepath": str(path), "duration": len(pcm) / rate, "text": "", "target_lang": "en-US"}) + "\n")
    for kind in ["manifest", "target_lang"]:
        try:
            model.eval()
            result = model.transcribe(str(manifest), batch_size=1, return_hypotheses=True) if kind == "manifest" else model.transcribe([str(path)], batch_size=1, return_hypotheses=True, target_lang="en-US")
            print(json.dumps({"kind": kind, "text": result[0].text, "tokens": result[0].y_sequence.tolist()}), flush=True)
        except Exception as error:
            print(json.dumps({"kind": kind, "exception": type(error).__name__, "message": str(error)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"exception": type(error).__name__, "message": str(error)}), file=sys.stderr)
        sys.exit(1)
