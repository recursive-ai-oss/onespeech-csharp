"""CPU/fp32 reference embeddings and config-declared feature outputs, no source inspection."""
import logging
import warnings
import argparse
import json
import sys
from pathlib import Path
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
ROOT = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache"); parser.add_argument("data"); parser.add_argument("ids"); parser.add_argument("output")
    parser.add_argument("start",type=int); parser.add_argument("count",type=int)
    parser.add_argument("--features",action="store_true")
    args=parser.parse_args()
    import torch
    import numpy as np
    import soundfile as sf
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(8)
    model=EncoderClassifier.from_hparams(source=ROOT,savedir=args.cache,run_opts={"device":"cpu"},overrides={"pretrained_path":ROOT})
    files={p.stem:p for split in ["test-clean","test-other"] for p in (Path(args.data)/split).rglob("*.flac")}
    ids=Path(args.ids).read_text().splitlines()[args.start:args.start+args.count]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    completed=0
    with torch.inference_mode():
        for utterance in ids:
            target=out/(utterance+".f32")
            if target.exists(): continue
            audio,rate=sf.read(files[utterance],dtype="float32",always_2d=True)
            if rate!=16000 or audio.shape[1]!=1: raise ValueError("Expected mono16k evaluation clip")
            signal=torch.from_numpy(audio[:,0].copy()).unsqueeze(0)
            embedding=model.encode_batch(signal,normalize=False).detach().cpu().numpy().reshape(-1)
            if not np.isfinite(embedding).all() or not np.linalg.norm(embedding)>0: raise ValueError("Invalid embedding")
            temporary=target.with_suffix(".partial"); embedding.astype("<f4").tofile(temporary); temporary.replace(target)
            if args.features:
                raw=model.mods.compute_features(signal)
                features=model.mods.mean_var_norm(raw.clone(),torch.ones(1))
                features.transpose(1,2).contiguous().numpy().astype("<f4").tofile(out/(utterance+".features.f32"))
            completed+=1
            print(json.dumps({"id":utterance,"dimension":int(embedding.size)}),flush=True)
    print(json.dumps({"completed":completed,"source":ROOT,"precision":"fp32","device":"cpu","normalizeEmbedding":False}),flush=True)

if __name__=="__main__":
    try: main()
    except Exception as error:
        print(json.dumps({"error":type(error).__name__,"message":str(error)[:1600]}),flush=True)
        sys.exit(1)
