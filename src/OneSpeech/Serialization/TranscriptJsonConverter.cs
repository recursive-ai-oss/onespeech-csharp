using System.Buffers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneSpeech.Serialization;

/// <summary>Uses the bounded version-1 Transcript wire format for System.Text.Json, including web endpoints. No reflection or serializer options can change that format.</summary>
public sealed class TranscriptJsonConverter : JsonConverter<Transcript>
{
    /// <summary>Creates the stateless converter.</summary>
    public TranscriptJsonConverter() { }
    /// <summary>Reads exactly one bounded schema envelope. Throws JsonException for malformed, unsupported or over-limit input.</summary>
    public override Transcript Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        // Check the byte bound before ParseValue allocates the DOM; JsonBounds then checks tokens, strings and depth.
        var probe = reader;
        long start = probe.TokenStartIndex;
        if (!probe.TrySkip() || probe.BytesConsumed - start > JsonBounds.MaxBytes) throw new JsonException("Transcript exceeds 8 MiB or is incomplete.");
        try
        {
            using var document = JsonDocument.ParseValue(ref reader);
            var buffer = new ArrayBufferWriter<byte>();
            using (var writer = new Utf8JsonWriter(buffer)) document.RootElement.WriteTo(writer);
            return TranscriptJson.Read(buffer.WrittenSpan);
        }
        catch (InvalidTranscriptJsonException e) { throw new JsonException(e.Message, e); }
    }
    /// <summary>Writes the canonical schema envelope, irrespective of naming and enum options. Throws JsonException for invalid or over-limit documents.</summary>
    public override void Write(Utf8JsonWriter writer, Transcript value, JsonSerializerOptions options)
    {
        try { writer.WriteRawValue(TranscriptJson.Write(value)); }
        catch (InvalidTranscriptJsonException e) { throw new JsonException(e.Message, e); }
    }
}
