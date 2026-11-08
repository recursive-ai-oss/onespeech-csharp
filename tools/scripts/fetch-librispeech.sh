#!/usr/bin/env bash
set -euo pipefail
# Archives are pinned independently of a mutable download page.
target=${1:?Usage: fetch-librispeech.sh TARGET}
mkdir -p "$target"
: > "$target/sources.tsv.partial"
cat > "$target/SHA256SUMS.expected" <<'SUMS'
39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23  test-clean.tar.gz
d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29  test-other.tar.gz
SUMS
for split in test-clean test-other; do
  file="$target/$split.tar.gz"
  url="https://www.openslr.org/resources/12/$split.tar.gz"
  if [[ ! -f "$file" ]]; then curl --fail --location --retry 3 "$url" -o "$file.partial"; mv "$file.partial" "$file"; fi
  printf '%s\t%s\t%s\n' "$url" 'CC-BY-4.0' "$split.tar.gz" >> "$target/sources.tsv.partial"
done
(cd "$target"; sha256sum -c SHA256SUMS.expected)
mv "$target/sources.tsv.partial" "$target/sources.tsv"
for split in test-clean test-other; do
  # Extract only the pinned corpus subtree, never arbitrary paths from an archive.
  if [[ ! -f "$target/.$split.extracted" ]]; then
    tar -xzf "$target/$split.tar.gz" -C "$target" --wildcards "LibriSpeech/$split/*"
    touch "$target/.$split.extracted"
  fi
done
printf '%s\n' 'LibriSpeech, OpenSLR 12. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/' > "$target/LICENSE-NOTE"
