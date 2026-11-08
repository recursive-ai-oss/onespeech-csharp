"""Execute the released CLI with captured diagnostics, never print module frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("environment", type=Path)
p.add_argument("assets", type=Path)
p.add_argument("split", type=Path)
a = p.parse_args()
for name, expected in [("english_mfa.zip", "a5c26c360b741e9a8e692416d942dae1fb2a8957dccc0826c4d6a87c98945f73"), ("english_mfa.dict", "30da2ad9168290da124369edb8011ebca5e9a6f0d4b0ad5e0fc33fa0fe12ca77")]:
    with (a.assets / name).open("rb") as asset:
        if hashlib.file_digest(asset, "sha256").hexdigest() != expected:
            raise SystemExit("MFA asset hash mismatch: " + name)
env = dict(os.environ, PYTHONWARNINGS="ignore", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MFA_ROOT_DIR=str(a.split / "work"))
env["PATH"] = str(a.environment / "bin") + os.pathsep + env["PATH"]
command = [str(a.environment / "bin/mfa"), "align", str(a.split / "corpus"), str(a.assets / "english_mfa.dict"), str(a.assets / "english_mfa.zip"), str(a.split / "truth"), "--output_format", "json", "--uses_speaker_adaptation", "false", "--num_jobs", "4", "--no_use_postgres", "--final_clean", "--temporary_directory", str(a.split / "work")]
(a.split / "command.json").write_text(json.dumps(command))
result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
# Save CLI messages only, excluding Python traceback/source lines conservatively.
lines = (result.stdout + "\n" + result.stderr).splitlines()
safe = [line for line in lines if line and not line.startswith((" ", "Traceback", "During handling", "The above exception"))]
(a.split / "mfa.log").write_text("\n".join(safe) + "\n")
print("MFA exit", result.returncode, "outputs", len(list((a.split / "truth").rglob("*.json"))))
raise SystemExit(result.returncode)
