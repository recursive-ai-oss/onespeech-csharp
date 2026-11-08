#!/usr/bin/env bash
set -euo pipefail
target=${1:?Usage: fetch-earnings22.sh TARGET}
pins=$(cd "$(dirname "$0")/pins" && pwd)
revision=c05ab6fd8b4b627d123c922a22a39e993dd37635
base="https://raw.githubusercontent.com/revdotcom/speech-datasets/$revision/earnings22"
mkdir -p "$target/media"
: > "$target/sources.tsv.partial"
while read -r hash name; do
  url="$base/media/$name"
  if [[ ! -f "$target/media/$name" ]]; then curl --fail --location --retry 3 "$url" -o "$target/media/$name.partial"; mv "$target/media/$name.partial" "$target/media/$name"; fi
  printf '%s\tCC-BY-SA-4.0\t%s\n' "$url" "media/$name" >> "$target/sources.tsv.partial"
done < "$pins/earnings22-media.sha256"
(cd "$target/media"; sha256sum -c "$pins/earnings22-media.sha256")
while IFS=$'\t' read -r hash name; do
  mkdir -p "$target/$(dirname "$name")"
  if [[ ! -f "$target/$name" ]]; then curl --fail --location --retry 3 "$base/$name" -o "$target/$name.partial"; mv "$target/$name.partial" "$target/$name"; fi
  # Check Git's content-addressed blob ID at the pinned commit before recording SHA-256.
  size=$(wc -c < "$target/$name")
  actual=$({ printf 'blob %s\0' "$size"; cat "$target/$name"; } | sha1sum | cut -d' ' -f1)
  [[ "$actual" == "$hash" ]] || { echo "Pinned content mismatch: $name" >&2; exit 1; }
  printf '%s\tCC-BY-SA-4.0\t%s\n' "$base/$name" "$name" >> "$target/sources.tsv.partial"
done < "$pins/earnings22-text.gitsha1"
(cd "$target"; { sed 's/  /  media\//' "$pins/earnings22-media.sha256"; cut -f2 "$pins/earnings22-text.gitsha1" | while read -r name; do sha256sum "$name"; done; } > SHA256SUMS.partial; mv SHA256SUMS.partial SHA256SUMS; sha256sum -c SHA256SUMS)
mv "$target/sources.tsv.partial" "$target/sources.tsv"
printf '%s\n' 'Earnings-22, Rev.com speech-datasets. CC BY-SA 4.0. Evaluation only; no redistribution.' > "$target/LICENSE-NOTE"
