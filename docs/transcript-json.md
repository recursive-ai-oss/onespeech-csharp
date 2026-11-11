# Transcript JSON, version 1

UTF-8 JSON, written and read using a generated System.Text.Json context. The root object has required `schemaVersion` (integer major version, currently **1**) and required `transcript`. Every other major version is rejected. Unknown fields are rejected; a future additive schema needs an explicit compatibility decision. Duplicate properties follow System.Text.Json's last-value behavior. Property names are case-sensitive.

All times are finite, nonnegative **seconds as JSON numbers**, relative to the beginning of the original input audio. Intervals are start-inclusive/end-exclusive; ends must not precede starts. Sub-tick precision rounds to the nearest .NET TimeSpan tick (100 ns), ties to even. Writers emit decimal seconds computed from integer ticks, preserving every tick on round-trip, including long durations. Values outside the TimeSpan range are rejected. Missing timing is represented by both `start` and `end` being null. No timing is inferred. Nulls are written explicitly. Text is preserved exactly.

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

Schema 1 supports **24 hours** of dense speech, with up to **600,000 words** (6.94 words/s over 24 h), **600,000 segments** and **600,000 turns**. Audio times and duration must be between 0 and 86,400 seconds. The wire repeats words inside segments and turns: the budget permits **1,800,000 total word entries across all `words` arrays**, so the root words can each appear once in a segment and once in a turn. Independent limits still apply; unusual verbosity or exceptionally long words can reach the byte budget first.

| Budget | Maximum |
|---|---:|
| Complete UTF-8 document | 1,000,000,000 bytes (1 GB) |
| Nesting depth | 32 |
| Tokens / combined object-array containers | 80,000,000 / 8,000,000 |
| Each words, segments or turns array | 600,000 items |
| Total elements across every array, including metadata | 3,200,000 |
| Each models or model-files array | 256 items |
| Transcript, segment or turn `text` string | 32 MiB raw encoded bytes |
| Any other string or property name | 64 KiB raw encoded bytes |

Raw string sizes include escapes. These budgets accommodate the ordinary 24 h dense document with three word representations rather than treating each copy as a new recording. Catalog JSON retains its separate 8 MiB / 64 KiB limits.

Both string and UTF-8 readers preflight **all wire budgets**, including per-array and aggregate counts, with a forward-only reader before allocating the document graph or decoded strings. The System.Text.Json converter follows the same preflight before source-generated deserialization, without an intermediate DOM. String callers necessarily already own their input; UTF-8 encoding allocation occurs only after exact byte-size validation. Fixed depth bookkeeping is bounded independently of input. Services should also enforce their own request-body limits appropriate to available memory; the library's maximum is not a recommendation to accept 1 GB network requests.

Invalid Unicode, malformed/truncated JSON, required-field omissions, null object/array members, invalid intervals/probabilities/provenance, unsupported versions or exceeded budgets throw `InvalidTranscriptJsonException` (or `JsonException` through the System.Text.Json converter). Empty arrays are allowed. Writers validate shape, count, duration and strings, then count encoded bytes through a bounded scratch writer before allocating the final JSON. The completed wire is preflighted, not deserialized into a second document. Schema version, fields, ordering and numeric encodings are unchanged.

Writers use the same validation and limits; callers should avoid constructing oversized documents. Segment/turn copies of words need not be object-identical to the top-level list. No cross-list identity, ordering, text reconstruction, or speaker attribution is invented by serialization.

## Round-trip example

This no-model example preserves an untimed word and its written form. Null speaker means speaker assignment was not processed; it differs from `SpeakerLabel.Unassigned` after an unsuccessful assignment.

```csharp
var document = new OneSpeech.Transcript
{
    Text = "Hello.",
    Language = "en",
    Words = [new OneSpeech.Word { Text = "Hello.", TimingMethod = OneSpeech.TimingMethod.None }]
};
string json = document.ToJson();
var restored = OneSpeech.Transcript.FromJson(json);
if (restored.Text != document.Text || restored.Words[0].Start is not null)
    throw new InvalidOperationException("Document round-trip changed text or invented timing.");
```

## System.Text.Json and web endpoints

`Transcript` carries a stateless converter. Plain `System.Text.Json` (including Web defaults) writes and reads this same **schema version 1** envelope, seconds and enum names; naming policies cannot produce a second shape. The documented wire format has not changed. Bounds and semantic validation are the same as `ToJson` / `FromJson`. The System.Text.Json adapter bounds the raw value to 8 MiB before a temporary raw JSON DOM is allocated; token, string and depth bounds are then checked before constructing the immutable transcript. Optional arrays omitted on the wire retain their documented empty defaults, including nested word/file arrays; explicit null arrays are refused. Invalid documents are `JsonException` through System.Text.Json and `InvalidTranscriptJsonException` through the explicit methods, with a field/index path when validation fails.

For NativeAOT or trimmed applications, include `Transcript` in the application's source-generated JSON context and use its `JsonTypeInfo` overload, also when registering an ASP.NET JSON resolver. Reflection-based serialization is not required by the library. If an application disables reflection without registering a context, System.Text.Json refuses the operation rather than falling back to another wire format.

```csharp
[JsonSerializable(typeof(Transcript))]
internal sealed partial class AppJsonContext : JsonSerializerContext;
// JsonSerializer.Serialize(transcript, AppJsonContext.Default.Transcript)
```

`Transcript`, `Segment`, `Turn` and `ModelInfo` compare their immutable collections by ordered element value. A valid document equals its JSON round trip and has the same hash code within the process. Equality does not ignore provenance, spacing or null timing.
