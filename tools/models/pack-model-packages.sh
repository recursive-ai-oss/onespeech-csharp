#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. All prepared files and packages stay outside it.
: "${ONESPEECH_ARTIFACTS:?Set ONESPEECH_ARTIFACTS to an external build directory}"
: "${MODEL_PACKAGE_CACHE:?Set MODEL_PACKAGE_CACHE to an external prepared-file cache}"
: "${MODEL_PACKAGE_FEED:?Set MODEL_PACKAGE_FEED to the destination local feed}"
dotnet build tools/OneSpeech.ModelPackages -c Release -m:4 -v quiet
preparer="$ONESPEECH_ARTIFACTS/bin/OneSpeech.ModelPackages/release/OneSpeech.ModelPackages.dll"
for spec in 'CamPlusEnglish|iic/speech_campplus_sv_en_voxceleb_16k' 'MoonshineStreamingTiny|moonshine-ai/moonshine-streaming-tiny' 'MoonshineTiny|moonshine-ai/moonshine-tiny' 'MoonshineBase|moonshine-ai/moonshine-base'; do
  name=${spec%%|*}; id=${spec#*|}; files="$MODEL_PACKAGE_CACHE/$name"
  dotnet "$preparer" "$id" "$files" "${MODEL_PUBLISHER_CACHE:-$MODEL_PACKAGE_CACHE}"
  dotnet pack "src/OneSpeech.Models.$name" -c Release -m:4 -v quiet -o "$MODEL_PACKAGE_FEED" -p:ModelPackageFiles="$files" -p:ModelPackagePreparer="$preparer"
  for package in "$MODEL_PACKAGE_FEED"/*.OneSpeech.Models."$name".*.nupkg; do
    size=$(wc -c < "$package")
    if [ "$size" -ge 250000000 ]; then echo "Model package exceeds 250 MB: $package" >&2; exit 1; fi
    echo "$package $size bytes"
  done
done
