using System.Collections.Immutable;
using OneSpeech.Serialization;

namespace OneSpeech;

/// <summary>How a word received its timing; None means no timing was obtained.</summary>
public enum TimingMethod
{
    /// <summary>No timing available.</summary>
    None,
    /// <summary>Timing supplied by the recognizer.</summary>
    Native,
    /// <summary>Timing from a forced aligner.</summary>
    ForcedAligner,
    /// <summary>Timing from a CTC trellis.</summary>
    CtcAligner,
    /// <summary>Timing from DTW alignment.</summary>
    Dtw
}

/// <summary>A document-local speaker identifier; not a biometric identity.</summary>
public sealed record SpeakerLabel
{
    /// <summary>Explicitly unassigned: no speaker turn covers the word.</summary>
    public static SpeakerLabel Unassigned { get; } = new() { Id = "unassigned" };
    /// <summary>Stable document-local ID. The reserved value unassigned means no covering turn.</summary>
    public required string Id { get; init; }
    /// <summary>Optional display name; null when unknown.</summary>
    public string? Name { get; init; }
}

/// <summary>A recognized word with timing provenance. Null timing is never estimated by this contract.</summary>
public sealed record Word
{
    /// <summary>Written form, including any desired spacing or punctuation.</summary>
    public required string Text { get; init; }
    /// <summary>Start on the audio clock; null if timing failed.</summary>
    public TimeSpan? Start { get; init; }
    /// <summary>End on the audio clock; null if timing failed.</summary>
    public TimeSpan? End { get; init; }
    /// <summary>Probability in [0,1], or null when unavailable.</summary>
    public double? Confidence { get; init; }
    /// <summary>Timing provenance; defaults to None.</summary>
    public TimingMethod TimingMethod { get; init; }
    /// <summary>Speaker, null if not processed, or Unassigned if no turn covered this word.</summary>
    public SpeakerLabel? Speaker { get; init; }
    /// <summary>Whether multiple speakers were active; false by default.</summary>
    public bool Overlap { get; init; }
    /// <summary>Language tag, or null when unknown.</summary>
    public string? Language { get; init; }
}

/// <summary>An immutable segment on the shared audio clock.</summary>
public sealed record Segment
{
    /// <summary>Compares all document fields and collection elements by value.</summary>
    public bool Equals(Segment? other) => ReferenceEquals(this, other) || other is not null && Text == other.Text && Start == other.Start && End == other.End && Language == other.Language && SequenceValue.Equals(Words, other.Words);
    /// <summary>Hashes the same fields and ordered collection elements used by equality.</summary>
    public override int GetHashCode() => HashCode.Combine(Text, Start, End, Language, SequenceValue.Hash(Words));

    /// <summary>Segment text; preserved exactly rather than reconstructed from words.</summary>
    public required string Text { get; init; }
    /// <summary>Start on the audio clock, or null when unavailable.</summary>
    public TimeSpan? Start { get; init; }
    /// <summary>End on the audio clock, or null when unavailable.</summary>
    public TimeSpan? End { get; init; }
    /// <summary>Language tag, or null when unknown.</summary>
    public string? Language { get; init; }
    /// <summary>Words in recognition order; empty when no word output exists.</summary>
    public ImmutableArray<Word> Words { get; init; } = [];
}

/// <summary>A speaker turn; turns may overlap.</summary>
public sealed record Turn
{
    /// <summary>Compares all document fields and collection elements by value.</summary>
    public bool Equals(Turn? other) => ReferenceEquals(this, other) || other is not null && Speaker == other.Speaker && Start == other.Start && End == other.End && Text == other.Text && SequenceValue.Equals(Words, other.Words);
    /// <summary>Hashes the same fields and ordered collection elements used by equality.</summary>
    public override int GetHashCode() => HashCode.Combine(Speaker, Start, End, Text, SequenceValue.Hash(Words));

    /// <summary>Speaker ID; Unassigned denotes no attribution.</summary>
    public required SpeakerLabel Speaker { get; init; }
    /// <summary>Inclusive turn start on the audio clock.</summary>
    public required TimeSpan Start { get; init; }
    /// <summary>Exclusive turn end on the audio clock.</summary>
    public required TimeSpan End { get; init; }
    /// <summary>Written turn text, preserved exactly.</summary>
    public required string Text { get; init; }
    /// <summary>Assigned words; empty by default.</summary>
    public ImmutableArray<Word> Words { get; init; } = [];
}

/// <summary>An immutable transcription document. Times are relative to the input audio, not wall time.</summary>
[System.Text.Json.Serialization.JsonConverter(typeof(TranscriptJsonConverter))]
public sealed record Transcript
{
    /// <summary>Compares all document fields and collection elements by value.</summary>
    public bool Equals(Transcript? other) => ReferenceEquals(this, other) || other is not null && Text == other.Text && Language == other.Language && Duration == other.Duration && SequenceValue.Equals(Words, other.Words) && SequenceValue.Equals(Segments, other.Segments) && SequenceValue.Equals(Turns, other.Turns) && SequenceValue.Equals(Models, other.Models);
    /// <summary>Hashes the same fields and ordered collection elements used by equality.</summary>
    public override int GetHashCode() => HashCode.Combine(Text, Language, Duration, SequenceValue.Hash(Words), SequenceValue.Hash(Segments), SequenceValue.Hash(Turns), SequenceValue.Hash(Models));

    /// <summary>Written text, preserved exactly, including whitespace.</summary>
    public required string Text { get; init; }
    /// <summary>Recognized words in input order; empty by default.</summary>
    public ImmutableArray<Word> Words { get; init; } = [];
    /// <summary>Recognition segments; empty by default.</summary>
    public ImmutableArray<Segment> Segments { get; init; } = [];
    /// <summary>Speaker turns; empty until diarization is performed.</summary>
    public ImmutableArray<Turn> Turns { get; init; } = [];
    /// <summary>Predominant language tag, or null if unknown.</summary>
    public string? Language { get; init; }
    /// <summary>Input duration, nonnegative, on the audio clock; zero by default.</summary>
    public TimeSpan Duration { get; init; }
    /// <summary>Models contributing to this document; empty when no model was used.</summary>
    public ImmutableArray<ModelInfo> Models { get; init; } = [];
    /// <summary>Returns Text unchanged; no normalization or spacing is applied.</summary>
    public string ToText() => Text;
    /// <summary>Writes schema version 1 JSON. Throws InvalidTranscriptJsonException for invalid or over-limit documents.</summary>
    public string ToJson() => TranscriptJson.Write(this);
    /// <summary>Reads bounded version 1 JSON. Throws InvalidTranscriptJsonException for malformed, invalid, or over-limit input.</summary>
    public static Transcript FromJson(string json) => TranscriptJson.Read(json);
    /// <summary>Reads bounded UTF-8 version 1 JSON. Throws InvalidTranscriptJsonException for invalid input.</summary>
    public static Transcript FromJson(ReadOnlySpan<byte> utf8Json) => TranscriptJson.Read(utf8Json);
}
