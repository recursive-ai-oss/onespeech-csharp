"""Deterministic synthetic feature probes against the observed public fbank invocation."""
import warnings
import logging
import sys
import json
from pathlib import Path
import subprocess
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

def main():
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.compliance.kaldi import fbank
    torch.set_num_threads(8)
    converter, output = sys.argv[1:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    tone = np.sin(np.arange(16001, dtype=np.float32) * (2*np.pi*440/16000)).astype(np.float32)*.1
    impulse = np.zeros(16001,dtype=np.float32); impulse[201] = 1
    probes = {"silence":np.zeros(16000,dtype=np.float32),"tone-odd":tone,"impulse":impulse,
              "dc-tone":tone+.2,"clipping":np.clip(tone*20,-1,1),
              "noise":rng.uniform(-.2,.2,16000).astype(np.float32),"exact-window":tone[:400],"two-window":tone[:560]}
    for name, audio in probes.items():
        wav = output/(name+".wav")
        sf.write(wav,audio,16000,subtype="FLOAT")
        with torch.inference_mode():
            reference = fbank(torch.from_numpy(audio).unsqueeze(0),num_mel_bins=80)
            reference = (reference-reference.mean(dim=0,keepdim=True)).T.contiguous().numpy().astype(np.float64).reshape(-1)
        target = output/name
        subprocess.run(["dotnet",converter,"speaker-probe","features",str(wav),str(target)],check=True,capture_output=True)
        ours = np.fromfile(target/"features.f32",dtype="<f4").astype(np.float64)
        diff = np.abs(ours-reference)
        print(json.dumps({"probe":name,"samples":int(audio.size),"maxAbs":float(diff.max()),"meanAbs":float(diff.mean())}),flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error":type(error).__name__,"message":str(error)[:1000]}),flush=True)
        sys.exit(1)
