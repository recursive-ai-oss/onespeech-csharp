using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using OneSpeech.Catalog;

namespace OneSpeech.Serialization;

internal static class JsonBounds
{
    internal const int MaxBytes = 8 * 1024 * 1024;
    internal const int MaxStringBytes = 64 * 1024;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    internal static void Check(ReadOnlySpan<byte> json)
    {
        if (json.Length > MaxBytes) throw new JsonException("JSON exceeds 8 MiB.");
        var reader = new Utf8JsonReader(json, new JsonReaderOptions { MaxDepth = 32 });
        int tokens = 0;
        int entries = 0;
        while (reader.Read())
        {
            if (++tokens > 500_000) throw new JsonException("Too many JSON tokens.");
            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray && ++entries > 100_000)
                throw new JsonException("Too many JSON containers.");
            if (reader.TokenType is JsonTokenType.String or JsonTokenType.PropertyName)
            {
                if (reader.ValueSpan.Length > MaxStringBytes) throw new JsonException("JSON string exceeds 64 KiB encoded.");
                try { _ = StrictUtf8.GetCharCount(reader.ValueSpan); }
                catch (DecoderFallbackException e) { throw new JsonException("Invalid UTF-8.", e); }
            }
        }
        if (tokens == 0) throw new JsonException("Empty JSON.");
    }
}

internal sealed record TranscriptEnvelope
{
    public required int SchemaVersion { get; init; }
    public required Transcript Transcript { get; init; }
}

internal static class TranscriptJson
{
    internal static Transcript Read(string json)
    {
        if (json is null || json.Length > JsonBounds.MaxBytes) throw new InvalidTranscriptJsonException("Missing or oversized JSON.");
        try
        {
            var encoding = new UTF8Encoding(false, true);
            if (encoding.GetByteCount(json) > JsonBounds.MaxBytes) throw new JsonException("JSON exceeds 8 MiB.");
            return Read(encoding.GetBytes(json));
        }
        catch (EncoderFallbackException e) { throw new InvalidTranscriptJsonException("Invalid Unicode in JSON.", e); }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript JSON.", e); }
    }

    internal static Transcript Read(ReadOnlySpan<byte> json)
    {
        try
        {
            JsonBounds.Check(json);
            var envelope = JsonSerializer.Deserialize(json, SpeechJsonContext.Default.TranscriptEnvelope)
                ?? throw new JsonException("Expected a transcript envelope.");
            if (envelope.SchemaVersion != 1) throw new JsonException("Unsupported transcript schema major version; expected 1.");
            Validate(envelope.Transcript);
            return envelope.Transcript;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript JSON: " + e.Message, e); }
    }

    internal static string Write(Transcript transcript)
    {
        try
        {
            Validate(transcript);
            string json = JsonSerializer.Serialize(new TranscriptEnvelope { SchemaVersion = 1, Transcript = transcript }, SpeechJsonContext.Default.TranscriptEnvelope);
            // The same limits apply in both directions so every accepted output can be read.
            _ = Read(json);
            return json;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript document.", e); }
    }

    private static void Require(bool valid)
    {
        if (!valid) throw new JsonException("Invalid transcript field or interval.");
    }

    private static void Interval(TimeSpan? start, TimeSpan? end)
    {
        Require(start.HasValue == end.HasValue);
        if (start.HasValue) Require(start.Value >= TimeSpan.Zero && end >= start);
    }

    private static void Speaker(SpeakerLabel? speaker)
    {
        if (speaker is not null) Require(!string.IsNullOrWhiteSpace(speaker.Id));
    }

    private static void Words(ImmutableArray<Word> words)
    {
        Require(!words.IsDefault);
        foreach (var word in words)
        {
            Require(word is not null);
            Require(word!.Text is not null && Enum.IsDefined(word.TimingMethod));
            Interval(word.Start, word.End);
            Require(word.Confidence is null || double.IsFinite(word.Confidence.Value) && word.Confidence is >= 0 and <= 1);
            Speaker(word.Speaker);
        }
    }

    private static void Validate(Transcript? transcript)
    {
        Require(transcript is not null);
        Require(transcript!.Text is not null && transcript.Duration >= TimeSpan.Zero);
        Require(!transcript.Segments.IsDefault && !transcript.Turns.IsDefault && !transcript.Models.IsDefault);
        Words(transcript.Words);
        foreach (var segment in transcript.Segments)
        {
            Require(segment is not null && segment.Text is not null);
            Interval(segment!.Start, segment.End);
            Words(segment.Words);
        }
        foreach (var turn in transcript.Turns)
        {
            Require(turn is not null && turn.Text is not null && turn.Speaker is not null);
            Speaker(turn!.Speaker);
            Interval(turn.Start, turn.End);
            Words(turn.Words);
        }
        foreach (var model in transcript.Models)
        {
            Require(model is not null);
            Require(!string.IsNullOrWhiteSpace(model!.Id) && !string.IsNullOrWhiteSpace(model.Source) &&
                !string.IsNullOrWhiteSpace(model.Revision) && !string.IsNullOrWhiteSpace(model.License) && model.Attribution is not null);
            Require(Enum.IsDefined(model.Tier) && !model.Files.IsDefault);
            foreach (var file in model.Files)
            {
                Require(file is not null && !string.IsNullOrWhiteSpace(file.Name) && file.Sha256 is { Length: 64 });
                Require(file!.Sha256.All(char.IsAsciiHexDigit));
            }
        }
    }
}

internal sealed class SecondsConverter : JsonConverter<TimeSpan>
{
    public override TimeSpan Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.Number || !reader.TryGetDecimal(out decimal seconds) || seconds < 0)
            throw new JsonException("Expected finite nonnegative seconds.");
        // Decimal seconds preserve the schema's 100-ns values without binary-float truncation.
        try { return TimeSpan.FromTicks(checked((long)decimal.Round(seconds * TimeSpan.TicksPerSecond, 0, MidpointRounding.ToEven))); }
        catch (OverflowException e) { throw new JsonException("Seconds exceed TimeSpan range.", e); }
    }
    public override void Write(Utf8JsonWriter writer, TimeSpan value, JsonSerializerOptions options) => writer.WriteNumberValue((decimal)value.Ticks / TimeSpan.TicksPerSecond);
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never, UseStringEnumConverter = true,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    Converters = new[] { typeof(SecondsConverter) })]
[JsonSerializable(typeof(TranscriptEnvelope))]
[JsonSerializable(typeof(CatalogEntry))]
internal sealed partial class SpeechJsonContext : JsonSerializerContext;
