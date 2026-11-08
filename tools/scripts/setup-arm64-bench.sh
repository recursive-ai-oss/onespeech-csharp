#!/usr/bin/env bash
set -euo pipefail
state=${1:?Usage: setup-arm64-bench.sh STATE_DIRECTORY}
mkdir -p "$state"/{sdk,tmp,data,logs,nuget,models,artifacts}
export TMPDIR="$state/tmp"
if [[ ! -x "$state/sdk/dotnet" ]]; then
  file=dotnet-sdk-10.0.403-osx-arm64.tar.gz
  curl --fail --location --retry 3 "https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.403/$file" -o "$state/tmp/$file"
  printf '%s  %s\n' 5893c66b48ea3674c7da18501059d606e96f3672fabe4d5e41762e692c7c83df3ee0b2b9dc5a0786191da2dc413d7335158101c14c935ab217f31ac2529770f3 "$state/tmp/$file" | shasum -a 512 -c -
  tar -xzf "$state/tmp/$file" -C "$state/sdk"
  rm "$state/tmp/$file"
fi
archive="$state/tmp/test-clean.tar.gz"
if [[ ! -d "$state/data/LibriSpeech/test-clean" ]]; then
  curl --fail --location --retry 3 https://www.openslr.org/resources/12/test-clean.tar.gz -o "$archive"
  printf '%s  %s\n' 39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 "$archive" | shasum -a 256 -c -
  tar -xzf "$archive" -C "$state/data" LibriSpeech/test-clean
  rm "$archive"
fi
meeting="$state/data/ES2004a.Mix-Headset.wav"
if [[ ! -f "$meeting" ]]; then
  curl --fail --location --retry 3 https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav -o "$meeting.partial"
  printf '%s  %s\n' 3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f "$meeting.partial" | shasum -a 256 -c -
  mv "$meeting.partial" "$meeting"
fi
printf '%s\n' 'https://www.openslr.org/resources/12/test-clean.tar.gz CC-BY-4.0 LibriSpeech, OpenSLR 12; derived prefixes/concatenation, no model training.' 'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav CC-BY-4.0 AMI Meeting Corpus, AMI Consortium; derived mono PCM.' > "$state/data/LICENSE-NOTE"
# Model acquisition is handled by --pull in the benchmark executable. It verifies
# catalog revision and SHA-256 and converts locally; do not copy another cache.
"$state/sdk/dotnet" --info
