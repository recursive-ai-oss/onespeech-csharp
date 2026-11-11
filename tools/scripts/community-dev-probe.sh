#!/usr/bin/env bash
# Records the community pipeline's intermediate artifacts on development-only audio (the first 600 s of the two
# development meetings), so clustering settings can be confirmed without touching any scored recording.
# Usage: community-dev-probe.sh PYANNOTE_PYTHON DATA_ROOT SNAPSHOT OUT_DIR
set -euo pipefail
py=${1:?}; data=${2:?}; snapshot=${3:?}; out=${4:?}
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
for m in ES2011a ES2011b; do
  "$py" -c "
import soundfile as sf
w, r = sf.read('$data/prepared/$m.wav', dtype='float32', always_2d=True)
sf.write('$out/$m-0-600.wav', w[:600 * r], r, subtype='PCM_16')
"
  wq run --class heavy --cores 8 --max 30m --label "onespeech u41 dev-only reference probe $m" -- bash -c \
    "ulimit -c 0; OMP_NUM_THREADS=8 '$py' '$repo/tools/oracles/pyannote/pipeline_probe.py' '$snapshot' '$out/$m-0-600.wav' '$out/out-$m' > '$out/$m.log' 2>&1"
done
