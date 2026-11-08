#!/usr/bin/env bash
set -euo pipefail
input=${1:?Usage: generate-casefold.sh CaseFolding-17.0.0.txt}
# Write to stdout. Input is Unicode data, never an implementation's source.
expected=ff8d8fefbf123574205085d6714c36149eb946d717a0c585c27f0f4ef58c4183
actual=$(sha256sum "$input" | cut -d' ' -f1)
[[ "$actual" == "$expected" ]] || { echo 'Unexpected CaseFolding data hash' >&2; exit 1; }
cat <<'HEADER'
#nullable enable
// Generated from Unicode 17.0.0 CaseFolding.txt (C and F mappings).
// https://www.unicode.org/Public/17.0.0/ucd/CaseFolding.txt
// Copyright © 2025 Unicode, Inc. See Unicode/LICENSE.txt (Unicode License V3).
using System.Text;
namespace OneSpeech.Evaluation;
internal static class CaseFolding
{
    internal static string Fold(string value)
    {
        var b=new StringBuilder();
        foreach(var r in value.EnumerateRunes())
            if(Mappings.TryGetValue(r.Value,out string? mapped))b.Append(mapped);else b.Append(r);
        return b.ToString();
    }
    private static readonly Dictionary<int,string> Mappings = new()
    {
HEADER
awk -F';' '$1 ~ /^[0-9A-F]+$/ && $2 ~ /[CF]/ { gsub(/ /,"",$1); n=split($3,a," "); v=""; for(i=1;i<=n;i++)if(a[i]!="") {p="00000000" a[i];v=v "\\U" substr(p,length(p)-7);} printf "        [0x%s]=\"%s\",\n",$1,v; }' "$input"
printf '    };\n}\n'
