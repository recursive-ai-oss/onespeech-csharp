using System.Collections.Immutable;
using System.Buffers;
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
    private static readonly UTF8Encoding WireEncoding = new(false, true);
    internal static Transcript Read(string json)
    {
        if (json is null || json.Length > TranscriptJsonBounds.MaxBytes) throw new InvalidTranscriptJsonException("Missing or oversized JSON.");
        try
        {
            var encoding = new UTF8Encoding(false, true);
            if (encoding.GetByteCount(json) > TranscriptJsonBounds.MaxBytes) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
            return Read(encoding.GetBytes(json));
        }
        catch (EncoderFallbackException e) { throw new InvalidTranscriptJsonException("Invalid Unicode in JSON.", e); }
        catch (ArgumentException e) { throw new InvalidTranscriptJsonException("Oversized or invalid Unicode in JSON.", e); }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript JSON.", e); }
    }

    internal static Transcript Read(ReadOnlySpan<byte> json)
    {
        try
        {
            TranscriptJsonBounds.Check(json);
            var envelope = JsonSerializer.Deserialize(json, SpeechJsonContext.Default.TranscriptEnvelope)
                ?? throw new JsonException("Expected a transcript envelope.");
            if (envelope.SchemaVersion != 1) throw new JsonException("Unsupported transcript schema major version; expected 1.");
            var transcript = (envelope.Transcript ?? throw new JsonException("Missing transcript payload at 'transcript'.")).ToTranscript();
            Validate(transcript);
            return transcript;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript JSON: " + e.Message, e); }
    }

    internal static Transcript Read(ref Utf8JsonReader reader)
    {
        try
        {
            var probe = reader;
            if (probe.TokenType == JsonTokenType.None && !probe.Read()) throw new JsonException("Empty JSON.");
            TranscriptJsonBounds.CheckValue(ref probe);
            var envelope = JsonSerializer.Deserialize(ref reader, SpeechJsonContext.Default.TranscriptEnvelope)
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
            var envelope = new TranscriptEnvelope { SchemaVersion = 1, Transcript = TranscriptPayload.From(transcript) };
            // Count the exact encoded wire size without retaining output. A caller
            // can share a large string across many words; count limits alone do
            // not bound the resulting JSON allocation.
            using (var budget = new Utf8JsonWriter(new JsonByteBudget()))
            {
                JsonSerializer.Serialize(budget, envelope, SpeechJsonContext.Default.TranscriptEnvelope);
                budget.Flush();
            }
            string json = JsonSerializer.Serialize(envelope, SpeechJsonContext.Default.TranscriptEnvelope);
            // Preflight the wire budgets without allocating a second document graph.
            var encoding = new UTF8Encoding(false, true);
            if (encoding.GetByteCount(json) > TranscriptJsonBounds.MaxBytes) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
            TranscriptJsonBounds.Check(encoding.GetBytes(json));
            return json;
        }
        catch (JsonException e) { throw new InvalidTranscriptJsonException("Invalid transcript document: " + e.Message, e); }
    }

    private sealed class JsonByteBudget : IBufferWriter<byte>
    {
        private long count;
        private byte[] buffer = new byte[16384];
        public void Advance(int bytes)
        {
            if (bytes < 0 || bytes > buffer.Length) throw new ArgumentOutOfRangeException(nameof(bytes));
            if (bytes > TranscriptJsonBounds.MaxBytes - count) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
            count += bytes;
        }
        public Memory<byte> GetMemory(int sizeHint = 0) { Ensure(sizeHint); return buffer; }
        public Span<byte> GetSpan(int sizeHint = 0) { Ensure(sizeHint); return buffer; }
        private void Ensure(int size)
        {
            if (size < 0 || size > TranscriptJsonBounds.MaxBytes) throw new JsonException("Transcript JSON exceeds 1 GB (1,000,000,000 bytes).");
            // Reuse a single token-sized buffer, never accumulate the whole wire.
            if (size > buffer.Length) buffer = new byte[size];
        }
    }

    private static void Text(string? value, string path, int limit = TranscriptJsonBounds.MaxStringBytes)
    {
        Require(value is not null, path);
        try { Require(WireEncoding.GetByteCount(value!) <= limit, path); }
        catch (EncoderFallbackException e) { throw new JsonException("Invalid Unicode at '" + path + "'.", e); }
    }

    private static void Require(bool valid, string path)
    {
        if (!valid) throw new JsonException($"Invalid transcript field '{path}'.");
    }

    private static void Interval(TimeSpan? start, TimeSpan? end, string path)
    {
        Require(start.HasValue == end.HasValue, path + ".start/end");
        if (start.HasValue) Require(start.Value >= TimeSpan.Zero && end >= start && end <= TimeSpan.FromSeconds(TranscriptJsonBounds.MaxSeconds), path + ".start/end");
    }

    private static void Speaker(SpeakerLabel? speaker, string path)
    {
        if (speaker is not null)
        {
            Require(!string.IsNullOrWhiteSpace(speaker.Id), path + ".id");
            Text(speaker.Id, path + ".id");
            if (speaker.Name is not null) Text(speaker.Name, path + ".name");
        }
    }

    private static void Words(ImmutableArray<Word> words, string path)
    {
        Require(!words.IsDefault && words.Length <= TranscriptJsonBounds.MaxItems, path);
        for (int i = 0; i < words.Length; i++)
        {
            var word = words[i];
            string item = $"{path}[{i}]";
            Require(word is not null, item);
            Text(word!.Text, item + ".text");
            if (word.Language is not null) Text(word.Language, item + ".language");
            Require(Enum.IsDefined(word.TimingMethod), item + ".timingMethod");
            Interval(word.Start, word.End, item);
            Require(word.Confidence is null || double.IsFinite(word.Confidence.Value) && word.Confidence is >= 0 and <= 1, item + ".confidence");
            Speaker(word.Speaker, item + ".speaker");
        }
    }

    private static void Validate(Transcript? transcript)
    {
        Require(transcript is not null, "transcript");
        Text(transcript!.Text, "transcript.text", TranscriptJsonBounds.MaxTextBytes);
        if (transcript.Language is not null) Text(transcript.Language, "transcript.language");
        Require(transcript.Duration >= TimeSpan.Zero && transcript.Duration <= TimeSpan.FromSeconds(TranscriptJsonBounds.MaxSeconds), "transcript.duration");
        Require(!transcript.Segments.IsDefault && transcript.Segments.Length <= TranscriptJsonBounds.MaxItems, "transcript.segments");
        Require(!transcript.Turns.IsDefault && transcript.Turns.Length <= TranscriptJsonBounds.MaxItems, "transcript.turns");
        Require(!transcript.Models.IsDefault && transcript.Models.Length <= 256, "transcript.models");
        Words(transcript.Words, "transcript.words");
        long wordCopies = transcript.Words.Length;
        for (int i = 0; i < transcript.Segments.Length; i++)
        {
            var segment = transcript.Segments[i]; string item = $"transcript.segments[{i}]";
            Require(segment is not null && segment.Text is not null, item + ".text");
            Text(segment!.Text, item + ".text", TranscriptJsonBounds.MaxTextBytes);
            if (segment.Language is not null) Text(segment.Language, item + ".language");
            Interval(segment.Start, segment.End, item);
            Words(segment.Words, item + ".words");
            wordCopies += segment.Words.Length;
            Require(wordCopies <= TranscriptJsonBounds.MaxWordCopies, "transcript.wordCopies");
        }
        for (int i = 0; i < transcript.Turns.Length; i++)
        {
            var turn = transcript.Turns[i]; string item = $"transcript.turns[{i}]";
            Require(turn is not null && turn.Text is not null && turn.Speaker is not null, item);
            Text(turn!.Text, item + ".text", TranscriptJsonBounds.MaxTextBytes);
            Speaker(turn.Speaker, item + ".speaker");
            Interval(turn.Start, turn.End, item);
            Words(turn.Words, item + ".words");
            wordCopies += turn.Words.Length;
            Require(wordCopies <= TranscriptJsonBounds.MaxWordCopies, "transcript.wordCopies");
        }
        for (int i = 0; i < transcript.Models.Length; i++)
        {
            var model = transcript.Models[i]; string item = $"transcript.models[{i}]";
            Require(model is not null, item);
            Require(!string.IsNullOrWhiteSpace(model!.Id), item + ".id");
            Require(!string.IsNullOrWhiteSpace(model.Source), item + ".source");
            Require(!string.IsNullOrWhiteSpace(model.Revision), item + ".revision");
            Require(!string.IsNullOrWhiteSpace(model.License), item + ".license");
            Text(model.Id, item + ".id"); Text(model.Source, item + ".source"); Text(model.Revision, item + ".revision");
            Text(model.License, item + ".license"); Text(model.Attribution, item + ".attribution");
            Require(Enum.IsDefined(model.Tier), item + ".tier");
            Require(!model.Files.IsDefault && model.Files.Length <= 256, item + ".files");
            for (int j = 0; j < model.Files.Length; j++)
            {
                var file = model.Files[j]; string filePath = $"{item}.files[{j}]";
                Require(file is not null && !string.IsNullOrWhiteSpace(file.Name), filePath + ".name");
                Text(file!.Name, filePath + ".name");
                Require(file.Sha256 is { Length: 64 } && file.Sha256.All(char.IsAsciiHexDigit), filePath + ".sha256");
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
