#!/usr/bin/env bash
# Run from the repository root, with .NET 10 on PATH. All outputs stay in the supplied directory.
set -euo pipefail
work=${1:?absolute scratch directory required}
seconds=${2:-120}
[[ "$work" = /* && "$seconds" =~ ^[0-9]+$ && "$seconds" -le 300 ]] || exit 2
mkdir -p "$work/tools" "$work/corpus" "$work/findings" "$work/pub"
curl -fsSL https://github.com/Metalnem/libfuzzer-dotnet/releases/download/v2025.05.02.0904/libfuzzer-dotnet-ubuntu -o "$work/libfuzzer-dotnet"
echo "c2c2a90d94c409a4af339a0d4f244e0442c5a5d249be0f1252ba07871f285958  $work/libfuzzer-dotnet" | sha256sum -c -
chmod +x "$work/libfuzzer-dotnet"
if [[ ! -x "$work/tools/sharpfuzz" ]]; then
  dotnet tool install SharpFuzz.CommandLine --version 2.2.0 --tool-path "$work/tools"
fi
dotnet publish fuzz/OneSpeech.Fuzz -c Release -o "$work/pub" -m:2 -v quiet
DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$work/pub/OneSpeech.dll"
cp fuzz/corpus/*.json "$work/corpus/"
if [[ "${ONESPEECH_FUZZ_CATALOG:-0}" == 1 ]]; then
  rm -f "$work/corpus/"*.json
  cp src/OneSpeech/Catalog/models.json "$work/corpus/catalog.json"
  echo '{"schemaVersion":1,"models":[]}' > "$work/corpus/empty.json"
fi
"$work/libfuzzer-dotnet" --target_path="$work/pub/OneSpeech.Fuzz" "$work/corpus" \
  -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 -max_len=65536 \
  -artifact_prefix="$work/findings/" -print_final_stats=1
