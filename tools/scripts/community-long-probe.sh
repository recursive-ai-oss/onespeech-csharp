#!/usr/bin/env bash
# Records the community pipeline's intermediate artifacts on one full long meeting, plus a short excerpt on the same
# device so a GPU run can be compared with the CPU reference first.
# Usage: community-long-probe.sh PYANNOTE_PYTHON WAV SNAPSHOT OUT_DIR [excerpt-only]
# Set ONESPEECH_ORACLE_DEVICE=cuda with the interpreter of tools/oracles/pyannote-gpu to run on the GPU.
set -euo pipefail
py=${1:?}; wav=${2:?}; snapshot=${3:?}; out=${4:?}; mode=${5:-full}
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
"$py" -c "
import soundfile as sf
w, r = sf.read('$wav', dtype='float32', always_2d=True)
sf.write('$out/excerpt-600-780.wav', w[600 * r:780 * r], r, subtype='PCM_16')
"
"$py" "$repo/tools/oracles/pyannote/pipeline_probe.py" "$snapshot" "$out/excerpt-600-780.wav" "$out/out-excerpt" > "$out/excerpt.log" 2>&1
[ "$mode" = excerpt-only ] && exit 0
"$py" "$repo/tools/oracles/pyannote/pipeline_probe.py" "$snapshot" "$wav" "$out/out-full" > "$out/full.log" 2>&1
