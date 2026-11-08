#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: fetch-ami-dev-channels.sh AMI_DATA_ROOT}
pins=$(cd "$(dirname "$0")/pins" && pwd)
mkdir -p "$target/downloads"
: > "$target/channels-sources.tsv.partial"
while read -r hash name; do
  id=${name%%.*}
  url="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$id/audio/$name"
  if [[ ! -f "$target/downloads/$name" ]]; then
    curl --fail --location --retry 3 --retry-all-errors --continue-at - "$url" -o "$target/downloads/$name.partial"
    mv "$target/downloads/$name.partial" "$target/downloads/$name"
  fi
  printf '%s\tCC-BY-4.0\t%s\t%s\n' "$url" "$hash" "downloads/$name" >> "$target/channels-sources.tsv.partial"
done < "$pins/ami-dev-headsets.sha256"
(cd "$target/downloads"; sha256sum -c "$pins/ami-dev-headsets.sha256")
mv "$target/channels-sources.tsv.partial" "$target/channels-sources.tsv"
printf '%s\n' 'AMI Meeting Corpus, AMI Consortium, CC BY 4.0. https://groups.inf.ed.ac.uk/ami/corpus/license.shtml' 'Development microphone-layout probes only: ES2011a/ES2011b Headset-0/1. Leakage cases are calculated on frame energy/cross-energy statistics, not rendered into new speech. No independent telephony accuracy claim.' > "$target/channels-LICENSE-NOTE"
