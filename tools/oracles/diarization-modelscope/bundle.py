"""Black-box ModelScope diarization bundle at the pins in diarization-bundle.tsv."""
import contextlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import sys
import warnings

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "4"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def denied(*_args, **_kwargs):
    raise RuntimeError("Network disabled; only pinned local models are permitted")


def main():
    cache, work, output, *audio = map(Path, sys.argv[1:])
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"oracle": "modelscope", "version": importlib.metadata.version("modelscope"),
              "precision": "fp32", "device": "cpu", "threads": 4,
              "bundle": "iic/speech_campplus_speaker-diarization_common",
              "revision": "0b056318fde24067871436d60d020fa01f3096b4", "network": "disabled"}
    if report["version"] != "1.40.2":
        raise RuntimeError("Unexpected oracle release")
    # Only configuration data is adapted: absolute local auxiliary paths prevent implicit downloads.
    root = cache / "diarization-reference"
    main_model = root / "speech_campplus_speaker-diarization_common"
    shutil.copytree(main_model, work / "bundle", dirs_exist_ok=True)
    camp = work / "camp"
    camp.mkdir(exist_ok=True)
    shutil.copyfile(cache / "speakers/camp-zh/campplus_cn_common.bin", camp / "campplus_cn_common.bin")
    # Use the publisher's pinned model configuration, fetched separately alongside the approved weight.
    shutil.copyfile(root / "camp-zh-configuration.json", camp / "configuration.json")
    config_file = work / "bundle/configuration.json"
    config = json.loads(config_file.read_text())
    config["model"].update(speaker_model=str(camp),
                           change_locator=str(root / "speech_campplus-transformer_scl_zh-cn_16k-common"),
                           vad_model=str(root / "speech_fsmn_vad_zh-cn-16k-common-pytorch"))
    config_file.write_text(json.dumps(config))
    try:
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        socket.create_connection = denied
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torch
            from modelscope.pipelines import pipeline
            torch.set_num_threads(4)
            diarizer = pipeline(task="speaker-diarization", model=str(work / "bundle"), device="cpu")
            outputs = []
            for wav in audio:
                with torch.inference_mode():
                    value = diarizer(str(wav))
                turns = value.get("text") if isinstance(value, dict) else None
                if not isinstance(turns, list):
                    raise ValueError("Public pipeline did not return a diarization turn list")
                recording = wav.stem
                (output.parent / (recording + ".turns.json")).write_text(json.dumps(turns, default=str))
                target = output.parent / (recording + ".reference.rttm")
                lines = []
                for turn in turns:
                    if not isinstance(turn, (list, tuple)) or len(turn) < 3:
                        raise ValueError("Unexpected diarization turn shape")
                    start, end, speaker = float(turn[0]), float(turn[1]), str(turn[2]).replace(" ", "_")
                    if end <= start or start < 0:
                        raise ValueError(f"Invalid public-pipeline time interval: start={start}, end={end}")
                    lines.append(f"SPEAKER {recording} 1 {start:.9f} {end-start:.9f} <NA> <NA> {speaker} <NA> <NA>\n")
                target.write_text("".join(lines), encoding="utf-8")
                outputs.append(str(target))
        report.update(status="ok", hypotheses=outputs)
    except Exception as error:
        report.update(status="unavailable", error=type(error).__name__, message=str(error)[:1200])
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)[:1200]}))
        sys.exit(2)
