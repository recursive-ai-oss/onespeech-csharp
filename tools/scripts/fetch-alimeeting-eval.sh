#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: fetch-alimeeting-eval.sh DATA_ROOT}
pins=$(cd "$(dirname "$0")/pins" && pwd)
mkdir -p "$target/downloads"
url=https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
name=Eval_Ali.tar.gz
if [[ ! -f "$target/downloads/$name" ]]; then
  curl --fail --location --retry 3 "$url" -o "$target/downloads/$name.partial"
  mv "$target/downloads/$name.partial" "$target/downloads/$name"
fi
(cd "$target/downloads"; sha256sum -c "$pins/alimeeting-eval.sha256")
read -r hash _ < "$pins/alimeeting-eval.sha256"
printf '%s\tCC-BY-SA-4.0\t%s\t%s\n' "$url" "$hash" "downloads/$name" > "$target/sources.tsv"
printf '%s\n' 'AliMeeting, Alibaba Group; Yu et al., M2MeT (ICASSP 2022). SLR119 Eval release, 8 meeting sessions.' 'CC BY-SA 4.0: https://www.openslr.org/119/' 'Evaluation only. Do not redistribute source or prepared data from this repository. Prepared slice: R8001_M8004_MS801 and R8003_M8001_MS801, far-field channel 1, nonempty TextGrid intervals; no fitting on Eval. Preserve original attribution and licence with all data.' > "$target/LICENSE-NOTE"
