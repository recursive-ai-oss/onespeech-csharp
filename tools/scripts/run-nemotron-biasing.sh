#!/usr/bin/env bash
set -euo pipefail
# Fixed Earnings-22 pair for Nemotron 3.5 ASR contextual biasing: baseline, phrases and absent-phrase control,
# offline (VAD-chunked) and through the live session. Each run is its own queue job and is skipped once its output exists.
# Arguments: dotnet-tool-dll (OneSpeech.BiasingEval) nemotron-converted-dir state-dir (after "prepare")
TOOL=$1 MODEL=$2 STATE=$3
for family in nemotron nemotron-live; do
  for case in 0 1; do
    for mode in baseline phrases control; do
      for attempt in 1 2; do
        set +e
        wq run --class heavy --cores 4 --max 30m --label "onespeech Nemotron biasing $family $case $mode" -- dotnet "$TOOL" run "$family" "$MODEL" "$STATE" "$case" "$mode"
        code=$?
        set -e
        ((code==0)) && break
        ((code==124)) || exit "$code"
      done
    done
  done
done
