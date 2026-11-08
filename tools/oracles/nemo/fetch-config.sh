#!/usr/bin/env bash
# Fetch only publisher model data. Never extract or open executable archive members.
set -euo pipefail
cache=${1:?Specify an external model cache directory}
mkdir -p "$cache"
revision=541d1f99c6b0c3cd0b11a95167540bb8edefd82b
archive="$cache/parakeet-tdt-0.6b-v3.nemo"
curl --fail --location --retry 3 "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/$revision/parakeet-tdt-0.6b-v3.nemo" -o "$archive.part"
echo "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d  $archive.part" | sha256sum --check
mv "$archive.part" "$archive"
tar -xOf "$archive" ./model_config.yaml > "$cache/model_config.yaml"
sha256sum "$cache/model_config.yaml"
