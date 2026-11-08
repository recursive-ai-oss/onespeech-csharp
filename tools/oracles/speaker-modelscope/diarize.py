"""Released ModelScope public-pipeline probe, restricted to an approved local checkpoint."""
import contextlib
import importlib.metadata
import io
import json
import logging
from pathlib import Path
import socket
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

def denied(*_args, **_kwargs):
    raise RuntimeError("Network disabled: this probe permits only the already verified English CAM++ checkpoint, not unchecked auxiliary downloads")

def main():
    model, output, *audio = sys.argv[1:]
    report = {"oracle": "modelscope", "version": importlib.metadata.version("modelscope"),
              "task": "speaker-diarization", "checkpoint": "iic/speech_campplus_sv_en_voxceleb_16k",
              "revision": "032b8131a7ad812f87061955ca974c99060c5a03", "precision": "fp32", "device": "cpu", "threads": 1, "network": "disabled"}
    if report["version"] != "1.40.2":
        raise RuntimeError("Unexpected oracle release")
    # Do not expose dependency logs or external exception frames. No captured text is inspected.
    try:
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        socket.create_connection = denied
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import torch
            from modelscope.pipelines import pipeline
            torch.set_num_threads(1)
            diarizer = pipeline(task="speaker-diarization", model=model, device="cpu")
            outputs = []
            for wav in audio:
                with torch.inference_mode():
                    value = diarizer(wav)
                turns = value.get("text") if isinstance(value, dict) else None
                if not isinstance(turns, list):
                    raise ValueError("Public pipeline did not return a diarization turn list")
                recording = Path(wav).stem
                target = str(Path(output).parent / (recording + ".reference.rttm"))
                lines = []
                for turn in turns:
                    if not isinstance(turn, (list, tuple)) or len(turn) < 3:
                        raise ValueError("Unexpected diarization turn shape")
                    start, end, speaker = float(turn[0]), float(turn[1]), str(turn[2]).replace(" ", "_")
                    if end <= start or start < 0:
                        raise ValueError("Invalid public-pipeline time interval")
                    lines.append(f"SPEAKER {recording} 1 {start:.9f} {end-start:.9f} <NA> <NA> {speaker} <NA> <NA>\n")
                Path(target).write_text("".join(lines), encoding="utf-8")
                outputs.append(target)
        report.update(status="ok", hypotheses=outputs)
    except Exception as error:
        report.update(status="unavailable", error=type(error).__name__, message=str(error)[:1200])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)[:1200]}))
        sys.exit(2)
