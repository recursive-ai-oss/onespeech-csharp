# Transcript JSON, version 1

UTF-8 JSON, written and read using a generated System.Text.Json context. The root object has required `schemaVersion` (integer major version, currently **1**) and required `transcript`. Every other major version is rejected. Unknown fields are rejected; a future additive schema needs an explicit compatibility decision. Duplicate properties follow System.Text.Json's last-value behavior. Property names are case-sensitive.

All times are finite, nonnegative **seconds as JSON numbers**, relative to the beginning of the original input audio. Intervals are start-inclusive/end-exclusive; ends must not precede starts. Sub-tick precision rounds to .NET TimeSpan precision (100 ns); large floating-point numbers have their usual precision limits. Missing timing is represented by both `start` and `end` being null. No timing is inferred. Nulls are written explicitly. Text is preserved exactly.

## Fields

`transcript`:
- `text`: required string, the complete written output (not reconstructed from words).
- `words`: ordered array of word objects, default `[]`.
- `segments`: ordered array of segment objects, default `[]`.
- `turns`: array of possibly overlapping speaker turns, default `[]`.
- `language`: language tag or null (default).
- `duration`: nonnegative audio duration in seconds, default 0.
- `models`: array of provenance objects, default `[]`.

Word:
- `text`: required string, including any supplied whitespace/punctuation.
- `start`, `end`: number or null (default null), either both present or both null.
- `confidence`: finite number in [0,1], or null (default).
- `timingMethod`: `None` (default), `Native`, `ForcedAligner`, `CtcAligner`, or `Dtw`; numeric representations are accepted only for defined values. Future methods require a schema/library update.
- `speaker`: speaker object or null; null means not processed; `{ "id": "unassigned", "name": null }` means processed but no turn covered the word.
- `overlap`: boolean, false by default.
- `language`: language tag or null (default).

Segment: required `text`; nullable `start`, `end`, `language`; `words` array, default `[]`.

Turn: required `speaker`, `start`, `end`, `text`; `words` array, default `[]`.

Speaker: required nonblank `id`, optional nullable `name`. IDs are local to the document, not biometric identities.

Model provenance: required nonblank `id`, `source` (upstream repo URL), `revision`, `license` (SPDX ID); required `tier` (`Tier1` or `Tier2`, or corresponding numeric 1/2), required `attribution` (exact text). `gated` defaults false. `files` defaults `[]`; each file has required nonblank `name` and a required 64-hex-digit `sha256`. Provenance records do not authorize downloading or redistribution.

## Reader limits and failures

Both string and UTF-8 readers enforce an 8 MiB UTF-8 input limit, maximum depth 32, maximum 500,000 tokens, maximum 100,000 combined object/array containers, and maximum 64 KiB raw encoded bytes per property name or string (including escapes). Thus primitive array counts are bounded by the token limit; object-array counts by both limits. Limits are checked with a forward-only reader before allocating the document graph or decoded strings. String callers necessarily already own their input; UTF-8 encoding allocation occurs only after byte-size validation. Invalid Unicode, malformed/truncated JSON, required-field omissions, null object/array members, invalid intervals/probabilities/provenance, and unsupported versions throw `InvalidTranscriptJsonException`. Empty arrays are allowed. These are document bounds, not unbounded-file ingestion APIs.

Writers use the same validation and limits; callers should avoid constructing oversized documents. Segment/turn copies of words need not be object-identical to the top-level list. No cross-list identity, ordering, text reconstruction, or speaker attribution is invented by serialization.
