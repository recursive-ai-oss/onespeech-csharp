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
dotnet publish fuzz/OneSpeech.AudioFuzz -c Release -o "$work/pub" -m:2 -v quiet
DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$work/pub/OneSpeech.dll"
for tag in 6 7; do printf "RIFF\\x26\\x00\\x00\\x00WAVEfmt \\x10\\x00\\x00\\x00$(printf "\\%03o" "$tag")\\x00\\x01\\x00\\x80\\x3e\\x00\\x00\\x80\\x3e\\x00\\x00\\x01\\x00\\x08\\x00data\\x02\\x00\\x00\\x00\\xff\\xff" > "$work/corpus/g711-$tag.wav"; done
"$work/libfuzzer-dotnet" --target_path="$work/pub/OneSpeech.AudioFuzz" "$work/corpus" \
  -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 -max_len=65536 \
  -artifact_prefix="$work/findings/" -print_final_stats=1
