namespace OneSpeech;

/// <summary>An immutable recognition update. All times are relative to the first input sample, never wall time.</summary>
public abstract record LiveEvent
{
    /// <summary>Audio accepted when this event became available. Nondecreasing within a session; includes endpoint silence.</summary>
    public required TimeSpan AudioTime { get; init; }
    /// <summary>Zero-based utterance number. An EndpointEvent closes this number; the next utterance increments it.</summary>
    public required long UtteranceId { get; init; }
}

/// <summary>A provisional segment replacing the previous PartialEvent for this utterance; never revises an earlier FinalEvent.</summary>
public sealed record PartialEvent : LiveEvent
{
    /// <summary>Revision starting at one and increasing strictly within an utterance.</summary>
    public required long Revision { get; init; }
    /// <summary>Current hypothesis on the session audio clock. Missing word timings remain null.</summary>
    public required Segment Segment { get; init; }
}

/// <summary>A committed segment. The session never changes or replaces it after emission.</summary>
public sealed record FinalEvent : LiveEvent
{
    /// <summary>FinalEvent hypothesis on the session audio clock, using the same Segment shape as file streaming.</summary>
    public required Segment Segment { get; init; }
    /// <summary>Committed speaker activity for this final, on the same audio clock; empty when diarization was not performed. May contain multiple speakers. Available even when word timing is missing; does not invent word-to-speaker assignments.</summary>
    public System.Collections.Immutable.ImmutableArray<Turn> SpeakerTurns { get; init; } = [];
}

/// <summary>Why an utterance closed, independent of whether speech produced any text.</summary>
public enum EndpointReason
{
    /// <summary>Enough silence followed speech.</summary>
    Silence,
    /// <summary>The configured maximum utterance length was reached; speech may continue in the next utterance.</summary>
    DurationLimit,
    /// <summary>The caller completed the input.</summary>
    EndOfInput
}

/// <summary>Closes an utterance after its FinalEvent, if any. Silence-only input need not emit an EndpointEvent.</summary>
public sealed record EndpointEvent : LiveEvent
{
    /// <summary>End of the retained audio, before endpoint silence. This is not an inferred word boundary.</summary>
    public required TimeSpan End { get; init; }
    /// <summary>Cause of closure.</summary>
    public required EndpointReason Reason { get; init; }
}

/// <summary>A speaker turn on the session audio clock. Does not revise previously emitted finals.</summary>
public sealed record SpeakerTurnEvent : LiveEvent
{
    /// <summary>Turn activity; recognition and diarization may finish at different audio times.</summary>
    public required Turn Turn { get; init; }
}
