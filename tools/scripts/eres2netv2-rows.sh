#!/usr/bin/env bash
# ERes2NetV2 rows on the worker: convert, parity on the fixed 50 clips, then trial embeddings in shards and scoring.
# usage: eres2netv2-rows.sh convert | ids <out> | reference <ids> <start> <count> <outdir> | ours <ids> <start> <count> <outdir> | parity <outours> <outref> | score <outours> <outref>
set -euo pipefail
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
export ONESPEECH_ARTIFACTS=~/onespeech/u42/artifacts
home=~/onespeech
data=$home/data/librispeech/LibriSpeech
checkpoint=$home/models/speakers/eres2netv2/pretrained_eres2netv2.ckpt
graph=$home/u42/eres2netv2-fp32.onemodel
python=$home/oracles/speaker-modelscope/bin/python
build() { dotnet build tools/OneSpeech.Convert -c Release -m:4 -v quiet -clp:NoSummary -p:SileroModelPath=$home/models/silero-vad/v6.2.3/silero_vad.onnx >&2; }
convert() { dotnet "$ONESPEECH_ARTIFACTS"/bin/OneSpeech.Convert/release/OneSpeech.Convert.dll "$@"; }
case "$1" in
  convert) build; convert speaker-convert-eres "$checkpoint" "$graph"; sha256sum "$graph"; stat -c %s "$graph" ;;
  reference) export PYTHONWARNINGS=ignore ERES_LOCAL=$home/u42/eres-local
    "$python" tools/oracles/speaker-modelscope/extract.py eres2netv2 "$data" "$2" "$5" "$3" "$4" $( [ "$4" -le 50 ] && echo --features ) | tail -3 ;;
  ours) build; convert speaker-extract-eres "$graph" "$data" "$2" "$5" "$3" "$4" | tail -3 ;;
  ids) build; convert speaker-ids eval/manifests/librispeech-speaker-trials.tsv "$data/test-clean" "$2" "$2.parity" ;;
  parity) "$python" tools/oracles/speaker-modelscope/parity.py eval/manifests/librispeech-speaker-parity50.txt "$2" "$3" ;;
  score) convert speaker-score eval/manifests/librispeech-speaker-trials.tsv "$2" "$3" "$4" ;;
esac
