#!/usr/bin/env bash
# Deterministic ID-only trials; input corpus is fetched separately under its own licence.
set -euo pipefail
clean=${1:?LibriSpeech test-clean directory required}
other=${2:?LibriSpeech test-other directory required}
output=${3:?output TSV path required}
seed=${4:-1729}
dotnet run --project tools/OneSpeech.Convert -- speaker-trials "$clean" "$other" "$output" "$seed"
