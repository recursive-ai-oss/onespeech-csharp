"""Numeric probes through the published model-card API and config-declared modules only."""
import warnings
import logging
import sys
import json
import hashlib
from pathlib import Path
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

def main():
    import numpy as np
    import torch
    import soundfile as sf
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(8)
    cache, destination = map(Path, sys.argv[1:])
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    classifier = EncoderClassifier.from_hparams(source="https://huggingface.co/"+SOURCE+"/resolve/"+REVISION, savedir=str(cache), run_opts={"device":"cpu"},
        overrides={"pretrained_path":"https://huggingface.co/"+SOURCE+"/resolve/"+REVISION})
    rng = np.random.default_rng(1729)
    tone = np.sin(np.arange(16001,dtype=np.float32)*(2*np.pi*440/16000)).astype(np.float32)*.1
    impulse = np.zeros(16001,dtype=np.float32); impulse[201]=1
    probes = {"silence":np.zeros(16000,dtype=np.float32),"tone":tone,"impulse":impulse,
        "noise":rng.uniform(-.2,.2,16000).astype(np.float32),"dc-tone":tone+.2,"short":tone[:401]}
    with torch.inference_mode():
        for name, audio in probes.items():
            signal = torch.from_numpy(audio).unsqueeze(0)
            # Names and callable objects come from the publisher's hyperparams.yaml modules data.
            raw = classifier.mods.compute_features(signal)
            normalized = classifier.mods.mean_var_norm(raw.clone(), torch.ones(1))
            embedding = classifier.encode_batch(signal, normalize=False) if audio.size >= 640 else None
            raw.transpose(1,2).contiguous().numpy().astype("<f4").tofile(destination/(name+".raw.f32"))
            normalized.transpose(1,2).contiguous().numpy().astype("<f4").tofile(destination/(name+".features.f32"))
            if embedding is not None:
                embedding.contiguous().numpy().astype("<f4").tofile(destination/(name+".embedding.f32"))
            sf.write(destination/(name+".wav"),audio,16000,subtype="FLOAT")
            print(json.dumps({"probe":name,"samples":int(audio.size),"features":list(raw.shape),"rawMin":float(raw.min()),"rawMax":float(raw.max()),"rawFirst":raw[0,0,:10].tolist(),"embeddingShape":list(embedding.shape) if embedding is not None else None}),flush=True)
    for name in ["embedding_model.ckpt","classifier.ckpt","mean_var_norm_emb.ckpt","hyperparams.yaml","label_encoder.txt"]:
        path = cache/name
        if path.exists():
            print(json.dumps({"file":name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}),flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error":type(error).__name__,"message":str(error)[:1600]}),flush=True)
        sys.exit(1)
