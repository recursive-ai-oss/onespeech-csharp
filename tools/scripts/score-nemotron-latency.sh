#!/usr/bin/env bash
# Scores one Nemotron streaming row: token identity against the reference outputs, then stable-word availability
# (the next word's first token publication time minus the word's native end) for the reference's own publication,
# ours, and optionally an earlier run.
# Usage: score-nemotron-latency.sh TOOL.dll CONVERTED_MODEL CORPUS_ROOT REFERENCE_ROOT OURS_ROOT PREVIOUS_ROOT|- CHUNK SPLIT
# Layout under each root: ours-SPLIT-live-CHUNK (ours, previous) or hf-SPLIT-CHUNK (reference), as written by
# run-nemotron-native-rows.sh and run-nemotron-rows.sh.
set -euo pipefail
T=$1 M=$2 CORPUS=$3 REF=$4 OURS=$5 PREV=$6 c=$7 s=$8
R=$REF/hf-$s-$c
echo "== chunk $c split $s"
dotnet "$T" nemotron-asr-eval "$CORPUS/test-$s" "$OURS/ours-$s-live-$c" "$R" "$OURS/score-$s-$c" | tail -1
echo "-- reference publication"; dotnet "$T" nemotron-asr-latency "$M" "$R" "$R" | tail -2
echo "-- ours"; dotnet "$T" nemotron-asr-latency "$M" "$R" "$OURS/ours-$s-live-$c" | tail -2
if [ "$PREV" != "-" ]; then echo "-- previous"; dotnet "$T" nemotron-asr-latency "$M" "$R" "$PREV/ours-$s-live-$c" | tail -2; fi
