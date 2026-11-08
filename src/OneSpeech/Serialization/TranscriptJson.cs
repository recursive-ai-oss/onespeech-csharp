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
    public required TranscriptPayload Transcript { get; init; }
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
            var transcript = (envelope.Transcript ?? throw new JsonException("Missing transcript payload at 'transcript'.")).ToTranscript();
            Validate(transcript);
            return transcript;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript JSON: " + e.Message, e); }
    }

    internal static string Write(Transcript transcript)
    {
        try
        {
            Validate(transcript);
            string json = JsonSerializer.Serialize(new TranscriptEnvelope { SchemaVersion = 1, Transcript = TranscriptPayload.From(transcript) }, SpeechJsonContext.Default.TranscriptEnvelope);
            // The same limits apply in both directions so every accepted output can be read.
            _ = Read(json);
            return json;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript document: " + e.Message, e); }
    }

    private static void Require(bool valid, string path)
    {
        if (!valid) throw new JsonException($"Invalid transcript field '{path}'.");
    }

    private static void Interval(TimeSpan? start, TimeSpan? end, string path)
    {
        Require(start.HasValue == end.HasValue, path + ".start/end");
        if (start.HasValue) Require(start.Value >= TimeSpan.Zero && end >= start, path + ".start/end");
    }

    private static void Speaker(SpeakerLabel? speaker, string path)
    {
        if (speaker is not null) Require(!string.IsNullOrWhiteSpace(speaker.Id), path + ".id");
    }

    private static void Words(ImmutableArray<Word> words, string path)
    {
        Require(!words.IsDefault, path);
        for (int i = 0; i < words.Length; i++)
        {
            var word = words[i];
            string item = $"{path}[{i}]";
            Require(word is not null, item);
            Require(word!.Text is not null, item + ".text");
            Require(Enum.IsDefined(word.TimingMethod), item + ".timingMethod");
            Interval(word.Start, word.End, item);
            Require(word.Confidence is null || double.IsFinite(word.Confidence.Value) && word.Confidence is >= 0 and <= 1, item + ".confidence");
            Speaker(word.Speaker, item + ".speaker");
        }
    }

    private static void Validate(Transcript? transcript)
    {
        Require(transcript is not null, "transcript");
        Require(transcript!.Text is not null, "transcript.text");
        Require(transcript.Duration >= TimeSpan.Zero, "transcript.duration");
        Require(!transcript.Segments.IsDefault, "transcript.segments");
        Require(!transcript.Turns.IsDefault, "transcript.turns");
        Require(!transcript.Models.IsDefault, "transcript.models");
        Words(transcript.Words, "transcript.words");
        for (int i = 0; i < transcript.Segments.Length; i++)
        {
            var segment = transcript.Segments[i]; string item = $"transcript.segments[{i}]";
            Require(segment is not null && segment.Text is not null, item + ".text");
            Interval(segment!.Start, segment.End, item);
            Words(segment.Words, item + ".words");
        }
        for (int i = 0; i < transcript.Turns.Length; i++)
        {
            var turn = transcript.Turns[i]; string item = $"transcript.turns[{i}]";
            Require(turn is not null && turn.Text is not null && turn.Speaker is not null, item);
            Speaker(turn!.Speaker, item + ".speaker");
            Interval(turn.Start, turn.End, item);
            Words(turn.Words, item + ".words");
        }
        for (int i = 0; i < transcript.Models.Length; i++)
        {
            var model = transcript.Models[i]; string item = $"transcript.models[{i}]";
            Require(model is not null, item);
            Require(!string.IsNullOrWhiteSpace(model!.Id), item + ".id");
            Require(!string.IsNullOrWhiteSpace(model.Source), item + ".source");
            Require(!string.IsNullOrWhiteSpace(model.Revision), item + ".revision");
            Require(!string.IsNullOrWhiteSpace(model.License), item + ".license");
            Require(model.Attribution is not null, item + ".attribution");
            Require(Enum.IsDefined(model.Tier), item + ".tier");
            Require(!model.Files.IsDefault, item + ".files");
            for (int j = 0; j < model.Files.Length; j++)
            {
                var file = model.Files[j]; string filePath = $"{item}.files[{j}]";
                Require(file is not null && !string.IsNullOrWhiteSpace(file.Name), filePath + ".name");
                Require(file!.Sha256 is { Length: 64 } && file.Sha256.All(char.IsAsciiHexDigit), filePath + ".sha256");
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
