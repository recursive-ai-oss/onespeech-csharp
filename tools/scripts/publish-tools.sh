#!/usr/bin/env bash
# Publishes the evaluation tools into one output directory: publish-tools.sh <output> <silero_vad.onnx>
set -euo pipefail
OUT=$1 SILERO=$2
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 MSBUILDDISABLENODEREUSE=1 DOTNET_PROCESSOR_COUNT=4
ART=${ONESPEECH_ARTIFACTS:-$OUT/art}
export ONESPEECH_ARTIFACTS=$ART
mkdir -p "$OUT"
for project in tools/OneSpeech.Convert tools/OneSpeech.WhisperProbe tools/OneSpeech.TimingFloor eval/OneSpeech.Eval; do
  name=$(basename "$project")
  nice dotnet publish "$project" -c Release -m:4 -v quiet -clp:NoSummary -p:SileroModelPath="$SILERO" -o "$OUT/$name"
done
