#!/usr/bin/env bash
# Bounded smoke for the character-vocabulary reader; all outputs stay outside the checkout.
set -euo pipefail
work=${1:?absolute scratch directory required}
seed=${2:?pinned tokenizer file required}
[[ "$work" = /* && -f "$seed" ]] || exit 2
mkdir -p "$work/tools" "$work/corpus" "$work/findings" "$work/pub"
curl -fsSL https://github.com/Metalnem/libfuzzer-dotnet/releases/download/v2025.05.02.0904/libfuzzer-dotnet-ubuntu -o "$work/libfuzzer-dotnet"
echo "c2c2a90d94c409a4af339a0d4f244e0442c5a5d249be0f1252ba07871f285958  $work/libfuzzer-dotnet" | sha256sum -c -
chmod +x "$work/libfuzzer-dotnet"
if [[ ! -x "$work/tools/sharpfuzz" ]]; then
  dotnet tool install SharpFuzz.CommandLine --version 2.2.0 --tool-path "$work/tools"
fi
dotnet publish fuzz/OneSpeech.CheckpointFuzz -c Release -o "$work/pub" -m:4 -v quiet
cp "$seed" "$work/corpus/vocabulary.model"
DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$work/pub/OneSpeech.dll"
ONESPEECH_SENTENCEPIECE=1 "$work/libfuzzer-dotnet" --target_path="$work/pub/OneSpeech.CheckpointFuzz" "$work/corpus" \
  -max_total_time=120 -timeout=10 -rss_limit_mb=2048 -max_len=131072 \
  -artifact_prefix="$work/findings/" -print_final_stats=1
