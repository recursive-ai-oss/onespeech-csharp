#!/usr/bin/env bash
set -euo pipefail
# Model cache directories already contain verified assets; make the catalog key
# point at them without copying or fetching another copy of the weights.
state=${1:?external state directory}
models=${2:?external model cache directory}
python3 - "$models" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
catalog = json.loads(pathlib.Path('src/OneSpeech/Catalog/models.json').read_text())
m = next(m for m in catalog['models'] if m['id'] == 'facebook/omniASR-CTC-300M')
key = hashlib.sha256((m['id'] + '\n' + m['revision'] + '\n' + '\n'.join(f['name'] + ':' + f['sha256'] for f in m['files'])).encode()).hexdigest()
target = root / key
if not target.exists():
    target.symlink_to(root / 'omniasr-ctc-300m', target_is_directory=True)
PY
for split in test-clean test-other; do
  for method in ctc dtw; do
    total=200; [[ $split == test-other ]] && total=100
    for ((begin=0; begin<total; begin+=25)); do
      wq run --class heavy --max 30m --cores 4 --label "onespeech timing $method $split $begin" -- \
        dotnet "$ONESPEECH_ARTIFACTS/bin/OneSpeech.TimingFloor/debug/OneSpeech.TimingFloor.dll" native "$method" "$state/$split" "$models" "$begin" 25
    done
  done
done
