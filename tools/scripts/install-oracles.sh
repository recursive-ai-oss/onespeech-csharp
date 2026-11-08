#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: install-oracles.sh TARGET}
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$target"
for name in openai-whisper transformers faster-whisper; do
  # Each environment is isolated, pinned by uv.lock, and always CPU-only.
  UV_PROJECT_ENVIRONMENT="$target/$name" uv sync --project "$root/eval/drivers/$name" --locked --python 3.13
  cp "$root/eval/drivers/$name/uv.lock" "$target/$name/installation.uv.lock"
  sha256sum "$target/$name/installation.uv.lock" > "$target/$name/installation.sha256"
done
