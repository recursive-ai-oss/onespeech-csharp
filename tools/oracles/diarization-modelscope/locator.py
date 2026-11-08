"""Probe only the locator's documented public pipeline; never inspect model objects."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import socket
import sys
import warnings

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "4"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def denied(*_args, **_kwargs):
    raise RuntimeError("Network disabled; only pinned local models are permitted")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    model, data, work, output = map(Path, sys.argv[1:])
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pins = {
        "campplus_cn_encoder.pt": "104c182d5865349a4c9a087e2adfe1b0e27f7e989e9b9ec1a9b9566b624413ef",
        "transformer_backend.pt": "dd9fb813a66927413fce473087326ca0c66aa049bc1d4fa3064f188fcb102a97",
        "configuration.json": "d1aaac1fe468a9efdbde606b69ab527146a651e4a91fb10b49a051f77ae01b2f",
    }
    for name, sha in pins.items():
        if digest(model / name) != sha:
            raise ValueError("Pinned model digest mismatch: " + name)
    version = importlib.metadata.version("modelscope")
    if version != "1.40.2":
        raise ValueError("Unexpected oracle release")
    report = {"oracle": "modelscope", "version": version, "precision": "fp32", "device": "cpu",
              "threads": 4, "revision": "c4e62f7b842dd4231b85adcbd7d5b587ed91a5c1",
              "model_sha256": pins, "network": "disabled", "probes": []}
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import numpy as np
        import soundfile as sf
        import torch
        from modelscope.pipelines import pipeline
        torch.set_num_threads(4)
        locator = pipeline(task="speaker-diarization", model=str(model), device="cpu")
        # Fixed offsets, not selected using speaker labels or model outcomes.
        sources = {
            "ES2011a": "130bc4218891a5b4cdbedcd716c94cd062f6ea13b2c19b77fe9d82e4023e1c78",
            "ES2011b": "63459ac9811903fe49f79982e9155b457425fa62c5f1f47d4047f512cd348c83",
        }
        for meeting, expected in sources.items():
            source = data / "prepared" / (meeting + ".wav")
            sha = digest(source)
            if sha != expected:
                raise ValueError("Pinned development audio digest mismatch: " + meeting)
            for start in (0, 60, 120):
                samples, rate = sf.read(source, start=start * 16000, frames=8 * 16000, dtype="float32")
                if rate != 16000 or samples.ndim != 1 or len(samples) != 128000:
                    raise ValueError("Expected prepared mono 16 kHz input")
                wav = work / f"{meeting}-{start}.wav"
                sf.write(wav, samples, rate, subtype="FLOAT")
                probe = {"id": wav.stem, "source_sha256": sha, "start_seconds": start,
                         "duration_seconds": 8,
                         "pcm_f32le_sha256": hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()}
                try:
                    with torch.inference_mode():
                        result = locator(str(wav))
                    # Only returned data, not attributes, hooks, state or internal calls.
                    probe["output"] = json.loads(json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))
                    probe["status"] = "ok"
                except Exception as error:
                    probe.update(status="unavailable", error=type(error).__name__, message=str(error)[:1200])
                report["probes"].append(probe)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)[:1200]}))
        sys.exit(2)
