using System.Buffers;
using System.Text;
using System.Text.Json;

namespace OneSpeech.Serialization;

// Schema 1 duplicates words in the document, segments and turns. Budget all three
// copies for 24 h at up to 600,000 words (6.94 words/s), not just the root list.
internal static class TranscriptJsonBounds
{
    // Below the runtime's maximum string length as well as a bounded UTF-8 budget.
    internal const int MaxBytes = 1_000_000_000;
    internal const int MaxTextBytes = 32 * 1024 * 1024;
    internal const int MaxStringBytes = 64 * 1024;
    internal const int MaxItems = 600_000;
    internal const int MaxWordCopies = 3 * MaxItems;
    internal const int MaxSeconds = 24 * 60 * 60;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    internal static void Check(ReadOnlySpan<byte> json)
    {
        if (json.Length > MaxBytes) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
        var reader = new Utf8JsonReader(json, new JsonReaderOptions { MaxDepth = 32 });
        if (!reader.Read()) throw new JsonException("Empty JSON.");
        CheckValue(ref reader);
        if (reader.Read()) throw new JsonException("Trailing JSON value.");
    }

    // Only fixed 33-slot bookkeeping for depth 32 is allocated before the forward-only scan.
    // No document nodes, decoded strings or attacker-sized arrays are allocated.
    internal static void CheckValue(ref Utf8JsonReader reader)
    {
        int initialDepth = reader.CurrentDepth;
        long start = reader.TokenStartIndex;
        var arrays = new int[33];
        var counts = new int[33];
        int tokens = 0, containers = 0, elements = 0, wordCopies = 0, pendingArray = 0;
        bool longText = false, duration = false;
        do
        {
            int depth = reader.CurrentDepth - initialDepth;
            if (depth is < 0 or > 32) throw new JsonException("Transcript JSON exceeds depth 32.");
            if (reader.BytesConsumed - start > MaxBytes) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
            if (++tokens > 80_000_000) throw new JsonException("Too many transcript JSON tokens.");
            var token = reader.TokenType;
            if (token is not (JsonTokenType.PropertyName or JsonTokenType.EndObject or JsonTokenType.EndArray) && depth > 0 && arrays[depth - 1] != 0)
            {
                int kind = arrays[depth - 1];
                int limit = kind is 4 or 5 ? 256 : MaxItems;
                if (++counts[depth - 1] > limit) throw new JsonException("Transcript JSON array exceeds its item limit.");
                if (++elements > 3_200_000) throw new JsonException("Too many transcript JSON array elements.");
                if (kind == 1 && ++wordCopies > MaxWordCopies) throw new JsonException("Too many transcript word copies.");
            }
            if (token is JsonTokenType.String or JsonTokenType.PropertyName)
            {
                long bytes = reader.HasValueSequence ? reader.ValueSequence.Length : reader.ValueSpan.Length;
                int limit = token == JsonTokenType.String && longText ? MaxTextBytes : MaxStringBytes;
                if (bytes > limit) throw new JsonException("Transcript JSON string exceeds its encoded byte limit.");
                try
                {
                    if (reader.HasValueSequence) _ = StrictUtf8.GetCharCount(reader.ValueSequence.ToArray());
                    else _ = StrictUtf8.GetCharCount(reader.ValueSpan);
                }
                catch (DecoderFallbackException e) { throw new JsonException("Invalid UTF-8.", e); }
            }
            if (token == JsonTokenType.PropertyName)
            {
                pendingArray = reader.ValueTextEquals("words") ? 1 : reader.ValueTextEquals("segments") ? 2 : reader.ValueTextEquals("turns") ? 3 : reader.ValueTextEquals("models") ? 4 : reader.ValueTextEquals("files") ? 5 : 6;
                longText = reader.ValueTextEquals("text") && (depth == 2 || (depth >= 2 && arrays[depth - 2] is 2 or 3));
                duration = reader.ValueTextEquals("start") || reader.ValueTextEquals("end") || (depth == 2 && reader.ValueTextEquals("duration"));
            }
            else
            {
                if (duration && token == JsonTokenType.Number && (!reader.TryGetDecimal(out decimal seconds) || seconds is < 0 or > MaxSeconds))
                    throw new JsonException("Transcript audio time exceeds 24 hours or is invalid.");
                longText = false; duration = false;
                if (token is JsonTokenType.StartObject or JsonTokenType.StartArray)
                {
                    if (depth == 32) throw new JsonException("Transcript JSON exceeds depth 32.");
                    if (++containers > 8_000_000) throw new JsonException("Too many transcript JSON containers.");
                    arrays[depth] = token == JsonTokenType.StartArray ? (pendingArray == 0 ? 6 : pendingArray) : 0;
                    counts[depth] = 0;
                }
                else if (token is JsonTokenType.EndObject or JsonTokenType.EndArray) arrays[depth] = 0;
                pendingArray = 0;
            }
            if (depth == 0 && token is not (JsonTokenType.StartObject or JsonTokenType.StartArray or JsonTokenType.PropertyName)) return;
        } while (reader.Read());
        throw new JsonException("Incomplete transcript JSON.");
    }
}
