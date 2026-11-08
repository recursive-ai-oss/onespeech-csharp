"""Black-box published pipeline probe. Never prints external exception frames."""
import logging
import warnings
import sys
import json
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

def describe(value):
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__, "value": str(value)[:160]}

def main():
    import numpy as np
    import torch
    import soundfile as sf
    import torchaudio.compliance.kaldi as kaldi
    observed = []
    public_fbank = kaldi.fbank
    def capture_fbank(waveform, *args, **kwargs):
        value = public_fbank(waveform, *args, **kwargs)
        observed.append((value.detach().cpu().clone(), {"kwargs": kwargs, "positional": str(args), "waveformMaxAbs": float(waveform.abs().max())}))
        return value
    # Pass-through capture at a documented public feature API boundary, no module inspection.
    kaldi.fbank = capture_fbank
    from modelscope.pipelines import pipeline
    torch.set_num_threads(8)
    model = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_en_voxceleb_16k", model_revision="032b8131a7ad812f87061955ca974c99060c5a03", device="cpu")
    signal = np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 440 / 16000)) * .1
    path = sys.argv[1]
    sf.write(path, signal, 16000, subtype="FLOAT")
    result = model([path, path])
    print(json.dumps({"pipeline": describe(result)}))
    # The standard Pipeline preprocess/forward contract exposes the computation stages.
    data = model.preprocess([path, path])
    print(json.dumps({"preprocess": describe(data)}))
    output = model.forward(data)
    print(json.dumps({"forward": describe(output)}))
    if observed:
        value, metadata = observed[0]
        value = value - value.mean(dim=0, keepdim=True)
        value.T.contiguous().numpy().astype("<f4").tofile(path + ".modelscope-features.f32")
        print(json.dumps({"observedPublicFbankCalls": len(observed), "firstCall": metadata}))
    from torchaudio.compliance.kaldi import fbank
    feat = fbank(torch.from_numpy(signal).unsqueeze(0), num_mel_bins=80, dither=0.0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat.T.contiguous().numpy().astype("<f4").tofile(path + ".features.f32")
    output.detach().cpu().numpy().astype("<f4").tofile(path + ".embedding.f32")
    for name, values in [("preprocess", data), ("forward", output)]:
        if isinstance(values, dict):
            for key, value in values.items():
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                if isinstance(value, np.ndarray):
                    np.save(path + "." + name + "." + str(key) + ".npy", value)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)[:1200]}))
        sys.exit(1)
