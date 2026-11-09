#!/usr/bin/env bash
# All tools, generated corpus files and build output live under the supplied scratch directory.
set -euo pipefail
if [[ -n "${DOTNET_ROOT:-}" ]]; then export DOTNET_ROOT_X64="$DOTNET_ROOT"; fi
work=${1:?absolute scratch directory required}
seconds=${2:-60}
[[ "$work" = /* && "$seconds" =~ ^[0-9]+$ && "$seconds" -ge 1 && "$seconds" -le 600 ]] || exit 2
# Publishing must start from an uninstrumented copy, including after a failed run.
rm -rf "$work/pub"
mkdir -p "$work/tools" "$work/corpus" "$work/findings" "$work/pub"
curl -fsSL https://github.com/Metalnem/libfuzzer-dotnet/releases/download/v2025.05.02.0904/libfuzzer-dotnet-ubuntu -o "$work/libfuzzer-dotnet"
echo "c2c2a90d94c409a4af339a0d4f244e0442c5a5d249be0f1252ba07871f285958  $work/libfuzzer-dotnet" | sha256sum -c -
chmod +x "$work/libfuzzer-dotnet"
if [[ ! -x "$work/tools/sharpfuzz" ]]; then
  dotnet tool install SharpFuzz.CommandLine --version 2.2.0 --tool-path "$work/tools"
fi
dotnet publish fuzz/OneSpeech.MoonshineFuzz -c Release -o "$work/pub" -m:4 -v quiet
"$work/pub/OneSpeech.MoonshineFuzz" --seed "$work/corpus"
DOTNET_ROLL_FORWARD=Major "$work/tools/sharpfuzz" "$work/pub/OneSpeech.dll"
"$work/libfuzzer-dotnet" --target_path="$work/pub/OneSpeech.MoonshineFuzz" "$work/corpus" \
  -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 -max_len=65536 \
  -artifact_prefix="$work/findings/" -print_final_stats=1
