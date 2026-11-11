#!/usr/bin/env bash
set -euo pipefail
# Full-corpus-ASR test partition: ES2004, IS1009, TS3003, EN2002.
# Publisher split: https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
root=${1:?data root}; tool=${2:?DiarizationData dll}
pins=$(cd "$(dirname "$0")/pins" && pwd)/ami-test.sha256
mkdir -p "$root/downloads" "$root/prepared"
annotations="$root/downloads/ami_public_manual_1.6.2.zip"
if [[ ! -f "$annotations" ]]; then
  curl -fsSL --retry 3 https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip -o "$annotations.partial"
  mv "$annotations.partial" "$annotations"
fi
printf '%s  %s\n' b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d "$annotations" | sha256sum -c -
printf '%s\n' 'AMI Meeting Corpus, AMI Consortium, CC-BY-4.0.' 'https://groups.inf.ed.ac.uk/ami/corpus/license.shtml' 'Manual annotations v1.6.2; Full-corpus-ASR test partition; Mix-Headset audio.' 'Derived mono PCM16 and RTTM (timed words, no gap bridging); evaluation only, retain attribution.' > "$root/LICENSE-NOTE"
for id in ES2004{a,b,c,d} IS1009{a,b,c,d} TS3003{a,b,c,d} EN2002{a,b,c,d}; do
  source="$root/downloads/$id.Mix-Headset.wav"
  url="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$id/audio/$id.Mix-Headset.wav"
  if [[ ! -f "$root/prepared/$id.manifest.jsonl" ]]; then
    curl -fsSL --retry 3 "$url" -o "$source.partial"
    mv "$source.partial" "$source"
    expected=$(awk -v name="$id.Mix-Headset.wav" '$2==name {print $1}' "$pins")
    [[ -n "$expected" ]]
    printf '%s  %s\n' "$expected" "$source" | sha256sum -c -
    printf '%s\tCC-BY-4.0\t%s\t%s\n' "$url" "$(sha256sum "$source" | cut -d' ' -f1)" "$id" >> "$root/sources.tsv"
    dotnet "$tool" ami "$root" "$id"
    cp "$root/manifest.jsonl" "$root/prepared/$id.manifest.jsonl"
    rm "$source"
  fi
done
cat "$root"/prepared/*.manifest.jsonl > "$root/manifest.jsonl"
