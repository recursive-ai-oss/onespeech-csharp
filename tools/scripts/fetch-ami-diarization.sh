#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: fetch-ami-diarization.sh DATA_ROOT}
pins=$(cd "$(dirname "$0")/pins" && pwd)
mkdir -p "$target/downloads"
: > "$target/sources.tsv.partial"
while read -r hash name; do
  if [[ "$name" = *.zip ]]; then
    url="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/$name"
  else
    id=${name%%.*}
    url="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$id/audio/$name"
  fi
  if [[ ! -f "$target/downloads/$name" ]]; then
    curl --fail --location --retry 3 "$url" -o "$target/downloads/$name.partial"
    mv "$target/downloads/$name.partial" "$target/downloads/$name"
  fi
  printf '%s\tCC-BY-4.0\t%s\t%s\n' "$url" "$hash" "downloads/$name" >> "$target/sources.tsv.partial"
done < "$pins/ami-diarization.sha256"
(cd "$target/downloads"; sha256sum -c "$pins/ami-diarization.sha256")
mv "$target/sources.tsv.partial" "$target/sources.tsv"
printf '%s\n' 'AMI Meeting Corpus, AMI Consortium. Manual annotations v1.6.2 (2017-04-10), Mix-Headset audio.' 'CC BY 4.0: https://groups.inf.ed.ac.uk/ami/corpus/license.shtml' 'Full-corpus-ASR dev slice ES2011a/ES2011b; test slice ES2004a/ES2004b. Derived mono PCM16/RTTM files are evaluation only and marked in manifest.jsonl. Preserve this attribution with the data.' > "$target/LICENSE-NOTE"
