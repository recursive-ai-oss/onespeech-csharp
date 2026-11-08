#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: check-short-diarization.sh EVAL_DLL LIBRISPEECH_TEST_CLEAN CAM_EN_GRAPH OUTPUT' >&2
  exit 2
fi
eval_dll=$1
corpus=$2
graph=$3
output=$4
mkdir -p "$output"
# Seed 1729 and selection rules are encoded in the C# preparer. Audio stays outside git.
dotnet "$eval_dll" short-diarization prepare-ls "$corpus" "$output/data"
dotnet "$eval_dll" short-diarization score "$graph" "$output/data/manifest.tsv" "$output/results.tsv" file
awk -F '\t' 'NR>1 {n++;correct+=($2==$3);error+=$5+$6+$7;reference+=$8} END {printf "count accuracy %d/%d; pooled DER %.4f%%\n",correct,n,100*error/reference}' "$output/results.tsv"
