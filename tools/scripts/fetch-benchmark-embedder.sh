#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: fetch-benchmark-embedder.sh TARGET_DIRECTORY}
mkdir -p "$target"
url='https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_en_voxceleb_16k/repo?Revision=032b8131a7ad812f87061955ca974c99060c5a03&FilePath=campplus_voxceleb.bin'
file="$target/campplus_voxceleb.bin"
if [[ ! -f "$file" ]]; then
  curl --fail --location --retry 3 --max-time 1200 "$url" -o "$file.partial"
  printf '%s  %s\n' 5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de "$file.partial" | shasum -a 256 -c -
  mv "$file.partial" "$file"
fi
printf '%s  %s\n' 5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de "$file" | shasum -a 256 -c -
printf '%s\n' "$url" 'Apache-2.0. CAM++ — Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng and Qian Chen. Published by 3D-Speaker/ModelScope iic. Retain upstream licence and notices. Unmodified publisher weights, local conversion by the benchmark runner.' > "$target/SOURCE-NOTE"
