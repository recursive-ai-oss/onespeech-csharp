using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>A diarizer that consumes consecutive PCM blocks without retaining the recording waveform.</summary>
public abstract class BlockDiarizer : Diarizer
{
    /// <summary>Contributing model provenance; empty when an implementation has none to declare.</summary>
    public virtual ImmutableArray<ModelInfo> Models => [];

    /// <summary>Returns globally consistent turns for consecutive blocks on a clock starting at zero. Blocks may reuse storage after enumeration advances.
    /// Implementations document rate/channel limits and retained embedding memory. Throws OperationCanceledException on cancellation.</summary>
    public abstract ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken);

    /// <summary>Diarizes one borrowed buffer on its own clock; cancellation and format limits follow DiarizeBlocks.</summary>
    public override ImmutableArray<Turn> Diarize(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        return DiarizeBlocks([audio], cancellationToken);
    }
}

/// <summary>Clustering strategy; neither strategy detects overlapping speakers.</summary>
public enum ClusteringMethod
{
    /// <summary>Eigengap-estimated spectral clustering for long recordings; cosine AHC for spans up to 60 seconds and numerical nonconvergence.</summary>
    Spectral,
    /// <summary>Average-link cosine agglomerative hierarchical clustering.</summary>
    Agglomerative
}

/// <summary>Offline clustering parameters; activity detection is supplied separately by the caller.</summary>
public sealed record ClusteringOptions
{
    /// <summary>Apply cosine resegmentation when no custom resegmenter or count hint is supplied, default true.
    /// Exact counts or MinSpeakers above 1 keep raw labels to avoid smoothing away a required speaker. Set false to disable default smoothing.</summary>
    public bool Resegment { get; init; } = true;
    /// <summary>Exact speaker count, or null to estimate. Range 1–32; must lie within MinSpeakers/MaxSpeakers and cannot exceed window count.</summary>
    public int? NumSpeakers { get; init; }
    /// <summary>Minimum count for nonempty speech, default 1, range 1–32. Insufficient windows produce a typed capability error.</summary>
    public int MinSpeakers { get; init; } = 1;
    /// <summary>Maximum speaker count, default 8, range MinSpeakers–32.</summary>
    public int MaxSpeakers { get; init; } = 8;
    /// <summary>Default Spectral. Recordings spanning at most 60 seconds use cosine AHC on full-duration windows;
    /// longer recordings use spectral clustering with AHC on numerical nonconvergence. Embedding failures always propagate.</summary>
    public ClusteringMethod Method { get; init; }
    /// <summary>Embedding duration in seconds, default 1.5, range 0.1–10.</summary>
    public double WindowSeconds { get; init; } = 1.5;
    // Internal ablation control. The public behavior retains the 100 ms minimum.
    internal double MinimumSegmentSeconds { get; init; } = 0.1;
    /// <summary>Window stride in seconds, default 0.25, range 0.01–WindowSeconds.</summary>
    public double ShiftSeconds { get; init; } = 0.25;
    /// <summary>AHC stopping cosine similarity, default 0.65 in [-1,1], ignored for an exact count. Not dev-fitted.</summary>
    public double AgglomerativeThreshold { get; init; } = 0.65;
    /// <summary>Stopping cosine similarity for the short-recording Spectral policy, default 0.3 in [-1,1].
    /// Ignored for an exact count or explicit Agglomerative method. Checked on AMI development excerpts.</summary>
    public double ShortRecordingThreshold { get; init; } = 0.3;
    /// <summary>Maximum retained embeddings, default 100,000, range 1–1,000,000. Exceeding it is refused rather than retaining unbounded data.</summary>
    public int MaxWindows { get; init; } = 100_000;
    /// <summary>Maximum uniformly spaced landmark windows used in dense clustering, default 128, range 16–256 and at least MaxSpeakers.</summary>
    public int MaxLandmarks { get; init; } = 128;

    internal void Validate()
    {
        if (MinSpeakers < 1 || MaxSpeakers < MinSpeakers || MaxSpeakers > 32 ||
            NumSpeakers is {} n && (n < MinSpeakers || n > MaxSpeakers) ||
            Method is not (ClusteringMethod.Spectral or ClusteringMethod.Agglomerative) || !double.IsFinite(WindowSeconds) || WindowSeconds < 0.1 || WindowSeconds > 10 ||
            !double.IsFinite(MinimumSegmentSeconds) || MinimumSegmentSeconds < 0.045 || MinimumSegmentSeconds > WindowSeconds ||
            !double.IsFinite(ShiftSeconds) || ShiftSeconds < 0.01 || ShiftSeconds > WindowSeconds ||
            !double.IsFinite(ShortRecordingThreshold) || ShortRecordingThreshold < -1 || ShortRecordingThreshold > 1 ||
            !double.IsFinite(AgglomerativeThreshold) || AgglomerativeThreshold < -1 || AgglomerativeThreshold > 1 ||
            MaxWindows < 1 || MaxWindows > 1_000_000 || MaxLandmarks < Math.Max(16, MaxSpeakers) || MaxLandmarks > 256)
            throw new ArgumentException("Invalid clustering options: check counts, finite durations, cosine threshold and landmark bound.");
    }
}

/// <summary>A speech-only embedding window on the recording clock. Embedding storage is borrowed only during resegmentation.</summary>
public sealed record EmbeddingWindow
{
    /// <summary>Inclusive start, in audio time.</summary>
    public required TimeSpan Start { get; init; }
    /// <summary>Exclusive end, in audio time.</summary>
    public required TimeSpan End { get; init; }
    /// <summary>Finite unit-length embedding. Do not mutate or retain its borrowed storage.</summary>
    public required ReadOnlyMemory<float> Embedding { get; init; }
}

/// <summary>Optional post-clustering stage for algorithmic resegmentation or a compatible published backend. The diarizer uses cosine resegmentation by default; ClusteringOptions.Resegment can disable it.</summary>
public abstract class EmbeddingResegmenter
{
    /// <summary>Returns one nonnegative label per window, using only labels present in initialLabels. Called synchronously before turn construction.
    /// Storage is borrowed for this call. Implementations must check cancellation and must not persist embeddings.</summary>
    public abstract int[] Resegment(IReadOnlyList<EmbeddingWindow> windows, ReadOnlySpan<int> initialLabels, CancellationToken cancellationToken);
}
