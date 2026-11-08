using System.Collections.Immutable;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneSpeech.Serialization;

// Mutable wire-only properties let the generated reader retain initializers for omitted fields.
// Public records remain immutable; their init-only members must not be populated with absent-array defaults.
internal sealed class TranscriptPayload
{
    [JsonRequired] public string Text { get; set; } = null!;
    public ImmutableArray<Word> Words { get; set; } = [];
    public ImmutableArray<SegmentPayload> Segments { get; set; } = [];
    public ImmutableArray<TurnPayload> Turns { get; set; } = [];
    public string? Language { get; set; }
    public TimeSpan Duration { get; set; }
    public ImmutableArray<ModelPayload> Models { get; set; } = [];

    internal Transcript ToTranscript()
    {
        RequireArray(Words, "transcript.words"); RequireArray(Segments, "transcript.segments");
        RequireArray(Turns, "transcript.turns"); RequireArray(Models, "transcript.models");
        return new() { Text = Text, Words = Words, Language = Language, Duration = Duration,
            Segments = [.. Segments.Select((s, i) => (s ?? throw Missing($"transcript.segments[{i}]")).ToSegment(i))],
            Turns = [.. Turns.Select((t, i) => (t ?? throw Missing($"transcript.turns[{i}]")).ToTurn(i))],
            Models = [.. Models.Select((m, i) => (m ?? throw Missing($"transcript.models[{i}]")).ToModel(i))] };
    }
    internal static TranscriptPayload From(Transcript t) => new()
    {
        Text = t.Text, Words = t.Words, Language = t.Language, Duration = t.Duration,
        Segments = [.. t.Segments.Select(s => new SegmentPayload { Text = s.Text, Start = s.Start, End = s.End, Language = s.Language, Words = s.Words })],
        Turns = [.. t.Turns.Select(s => new TurnPayload { Speaker = s.Speaker, Start = s.Start, End = s.End, Text = s.Text, Words = s.Words })],
        Models = [.. t.Models.Select(m => new ModelPayload { Id = m.Id, Source = m.Source, Revision = m.Revision, Files = m.Files, Tier = m.Tier, License = m.License, Attribution = m.Attribution, Gated = m.Gated })]
    };
    internal static JsonException Missing(string path) => new($"Invalid or null transcript field '{path}'.");
    internal static void RequireArray<T>(ImmutableArray<T> values, string path)
    {
        if (values.IsDefault) throw Missing(path);
    }
}

internal sealed class SegmentPayload
{
    [JsonRequired] public string Text { get; set; } = null!;
    public TimeSpan? Start { get; set; }
    public TimeSpan? End { get; set; }
    public string? Language { get; set; }
    public ImmutableArray<Word> Words { get; set; } = [];
    internal Segment ToSegment(int index)
    {
        TranscriptPayload.RequireArray(Words, $"transcript.segments[{index}].words");
        return new() { Text = Text, Start = Start, End = End, Language = Language, Words = Words };
    }
}

internal sealed class TurnPayload
{
    [JsonRequired] public SpeakerLabel Speaker { get; set; } = null!;
    [JsonRequired] public TimeSpan Start { get; set; }
    [JsonRequired] public TimeSpan End { get; set; }
    [JsonRequired] public string Text { get; set; } = null!;
    public ImmutableArray<Word> Words { get; set; } = [];
    internal Turn ToTurn(int index)
    {
        TranscriptPayload.RequireArray(Words, $"transcript.turns[{index}].words");
        return new() { Speaker = Speaker, Start = Start, End = End, Text = Text, Words = Words };
    }
}

internal sealed class ModelPayload
{
    [JsonRequired] public string Id { get; set; } = null!;
    [JsonRequired] public string Source { get; set; } = null!;
    [JsonRequired] public string Revision { get; set; } = null!;
    public ImmutableArray<ModelFileInfo> Files { get; set; } = [];
    [JsonRequired] public ModelTier Tier { get; set; }
    [JsonRequired] public string License { get; set; } = null!;
    [JsonRequired] public string Attribution { get; set; } = null!;
    public bool Gated { get; set; }
    internal ModelInfo ToModel(int index)
    {
        TranscriptPayload.RequireArray(Files, $"transcript.models[{index}].files");
        return new() { Id = Id, Source = Source, Revision = Revision, Files = Files, Tier = Tier, License = License, Attribution = Attribution, Gated = Gated };
    }
}
