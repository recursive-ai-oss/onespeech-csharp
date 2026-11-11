#!/usr/bin/env bash
set -euo pipefail
# Run one fixed-policy system, one independently bounded reservation per recording.
kind=${1:?default|modelscope|nemotron|pyannote|community|communitynative}
data=${2:?prepared data root}; out=${3:?output root}
: "${EVAL_DLL:?}" "${CONVERT_DLL:?}" "${MODEL_ROOT:?}" "${MODEL_CACHE:?}" "${CAMP_GRAPH:?}" "${CAMP_ID:?}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
for wav in "$data"/prepared/*.wav; do
  id=$(basename "$wav" .wav)
  if [[ "$kind" == nemotron ]]; then
    [[ ! -f "$out/$id.metrics.json" ]] || continue
  else
    [[ ! -f "$out/$id.rttm" ]] || continue
  fi
  case "$kind" in
    default) command=(dotnet "$EVAL_DLL" diarize file-default "$CAMP_GRAPH" "$wav" "$out/$id.rttm"); cores=4 ;;
    pyannote) command=(dotnet "$CONVERT_DLL" pyannote-diarize "$MODEL_ROOT/pyannote-gated/pyannote/segmentation-3.0" "$MODEL_CACHE" "$CAMP_ID" "$wav" "$out/$id.rttm"); cores=4 ;;
    community) command=("${PYANNOTE_PYTHON:?}" "$repo/tools/oracles/pyannote/diarize.py" "$MODEL_ROOT/pyannote-gated/pyannote/speaker-diarization-community-1" "$wav" "$out/$id.rttm" --threads 8); cores=8 ;;
    communitynative) command=(dotnet "$CONVERT_DLL" community-diarize "$MODEL_ROOT/pyannote-gated/pyannote/speaker-diarization-community-1" "$MODEL_CACHE" "$wav" "$out/$id.rttm" 8); cores=8 ;;
    modelscope) command=("${MODELSCOPE_PYTHON:?}" "$repo/tools/oracles/diarization-modelscope/bundle.py" "$MODEL_ROOT" "$out/work" "$out/$id.report.json" "$wav"); cores=4 ;;
    nemotron) command=(bash -c 'set -e; "$1" "$2" "$3" "$4" "$5"; dotnet "$6" nemotron-native "$9" "$4" "$7" "$8" "$5/$(basename "$4" .wav).json"' _ "${TRANSFORMERS_PYTHON:?}" "$repo/tools/oracles/nemotron/reference.py" "${NEMOTRON_MODEL:?}" "$wav" "$out/reference" "$CONVERT_DLL" "$out" "$data/prepared/$id.rttm" "${NEMOTRON_RUNTIME:?resolved directory containing publisher weights and runtime graphs}"); cores=4 ;;
    *) exit 2 ;;
  esac
  wq run --class heavy --cores "$cores" --max 30m --label "onespeech diarization $kind $id" -- bash -c 'ulimit -c 0; cat /proc/loadavg; "$@"; code=$?; cat /proc/loadavg; exit "$code"' _ "${command[@]}" > "$out/$id.log" 2>&1 || { echo "$kind $id failed; see recording log"; continue; }
  if [[ "$kind" == modelscope && -f "$out/$id.reference.rttm" ]]; then mv "$out/$id.reference.rttm" "$out/$id.rttm"; fi
  if [[ -f "$out/$id.rttm" ]]; then echo "$kind $id completed"; else echo "$kind $id failed: missing RTTM"; fi
done
