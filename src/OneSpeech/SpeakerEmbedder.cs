namespace OneSpeech;

/// <summary>Extracts speaker identity vectors synchronously on the CPU. Implementations own their model resources.</summary>
public abstract class SpeakerEmbedder : IDisposable
{
    /// <summary>Checkpoint identity, precision and licence provenance.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Number of elements in each embedding; always positive.</summary>
    public abstract int Dimension { get; }
    /// <summary>Required mono input sample rate in Hz; implementations reject other rates and multiple channels.</summary>
    public abstract int SampleRate { get; }
    /// <summary>Extracts a finite, nonzero vector of <see cref="Dimension"/> elements into caller-owned storage.
    /// Input must be nonempty mono PCM at <see cref="SampleRate"/> Hz. Destination length must equal Dimension.
    /// Throws ArgumentException for unsupported input or storage, OperationCanceledException on cancellation,
    /// and ObjectDisposedException after disposal. No audio or embedding is persisted.</summary>
    public abstract void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken);
    /// <summary>Releases owned model resources. Repeated calls are harmless.</summary>
    public abstract void Dispose();
}
