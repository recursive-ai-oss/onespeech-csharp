using System.Collections.Immutable;

namespace OneSpeech.Whisper;

/// <summary>Whisper's task. Translation produces English text from supported source languages.</summary>
public enum WhisperTask
{
    /// <summary>Transcribe in the source language.</summary>
    Transcribe,
    /// <summary>Translate speech into English.</summary>
    Translate
}

/// <summary>Offline Whisper decoding settings. Defaults follow the released reference's public transcribe API, not its CLI beam-search defaults.</summary>
public sealed record WhisperOptions
{
    /// <summary>Transcribe by default.</summary>
    public WhisperTask Task { get; init; }
    /// <summary>Fallback temperatures, in order; defaults to 0, .2, .4, .6, .8, 1. Zero is greedy; positive values sample one candidate.</summary>
    public ImmutableArray<double> Temperatures { get; init; } = [0, .2, .4, .6, .8, 1];
    /// <summary>Retry above this UTF-8/zlib compression ratio; 2.4 by default, null disables the test.</summary>
    public double? CompressionRatioThreshold { get; init; } = 2.4;
    /// <summary>Retry below this mean token log probability; -1 by default, null disables the test.</summary>
    public double? LogProbabilityThreshold { get; init; } = -1;
    /// <summary>Skip high-no-speech windows unless their mean token log probability exceeds the threshold; .6 by default, null disables skipping.</summary>
    public double? NoSpeechThreshold { get; init; } = .6;
    /// <summary>Maximum initial timestamp in seconds; 1 by default, range 0..30.</summary>
    public double MaxInitialTimestamp { get; init; } = 1;
    /// <summary>Condition on previous windows by default. A fallback above .5 resets context for the next window.</summary>
    public bool ConditionOnPreviousText { get; init; } = true;
    /// <summary>Compute DTW word times by default. A failed alignment leaves both endpoints null.</summary>
    public bool WordTimestamps { get; init; } = true;
    /// <summary>Maximum generated tokens per attempt, 224 by default; range 1..224.</summary>
    public int MaxTokens { get; init; } = 224;
    /// <summary>Optional beam width at temperature zero, 1..16. Null (default) selects greedy decoding; positive-temperature fallback samples one candidate.</summary>
    public int? BeamSize { get; init; }
    /// <summary>OneModel intra-op threads, 1 by default; range 1..256.</summary>
    public int Threads { get; init; } = 1;
    /// <summary>Deterministic sampling seed, 0 by default. Only positive-temperature fallback uses it.</summary>
    public uint SamplingSeed { get; init; }
    /// <summary>Optional progress observer in processed audio time, called after each window.</summary>
    public IProgress<TimeSpan>? Progress { get; init; }

    internal void Validate()
    {
        if (BeamSize is < 1 or > 16) throw new ArgumentOutOfRangeException(nameof(BeamSize), BeamSize, "Whisper BeamSize must be 1–16 or null.");
        if (Threads is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(Threads), Threads, "Whisper Threads must be 1–256.");
        if (MaxTokens is < 1 or > 224) throw new ArgumentOutOfRangeException(nameof(MaxTokens), MaxTokens, "Whisper MaxTokens must be 1–224.");
        if (!Enum.IsDefined(Task)) throw new ArgumentOutOfRangeException(nameof(Task), Task, "Unknown Whisper Task.");
        if (Temperatures.IsDefaultOrEmpty || Temperatures.Length > 16 || Temperatures.Any(t => !double.IsFinite(t) || t < 0 || t > 2)) throw new ArgumentException("Whisper Temperatures requires 1–16 finite values in [0,2].", nameof(Temperatures));
        if (!double.IsFinite(MaxInitialTimestamp) || MaxInitialTimestamp is < 0 or > 30) throw new ArgumentOutOfRangeException(nameof(MaxInitialTimestamp), MaxInitialTimestamp, "Whisper MaxInitialTimestamp must be 0–30 seconds.");
        if (CompressionRatioThreshold is { } compression && (!double.IsFinite(compression) || compression <= 0)) throw new ArgumentOutOfRangeException(nameof(CompressionRatioThreshold), compression, "Whisper CompressionRatioThreshold must be finite and positive, or null.");
        if (LogProbabilityThreshold is { } probability && !double.IsFinite(probability)) throw new ArgumentOutOfRangeException(nameof(LogProbabilityThreshold), probability, "Whisper LogProbabilityThreshold must be finite, or null.");
        if (NoSpeechThreshold is { } silence && (!double.IsFinite(silence) || silence is < 0 or > 1)) throw new ArgumentOutOfRangeException(nameof(NoSpeechThreshold), silence, "Whisper NoSpeechThreshold must be in [0,1], or null.");
    }
}
