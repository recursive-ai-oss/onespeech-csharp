namespace OneSpeech;

/// <summary>Interleaved normalized float PCM. Caller owns storage and must not mutate it during processing. Roles reject incompatible rates/channels or nonfinite samples with ArgumentException; corrupt encoded files instead produce UnsupportedAudioException at decoding.</summary>
public sealed class AudioBuffer
{
    /// <summary>Constructs a PCM view. Sample rate and channel count must be positive; sample count must be divisible by channels.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Rate or channels is nonpositive.</exception>
    /// <exception cref="ArgumentException">Samples do not contain whole interleaved frames.</exception>
    public AudioBuffer(ReadOnlyMemory<float> samples, int sampleRate, int channels = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        if (samples.Length % channels != 0) throw new ArgumentException("PCM must contain whole interleaved frames.", nameof(samples));
        Samples = samples;
        SampleRate = sampleRate;
        Channels = channels;
    }
    /// <summary>Interleaved PCM; mono has one sample per frame. Empty input is allowed.</summary>
    public ReadOnlyMemory<float> Samples { get; }
    /// <summary>Samples per second per channel, in Hz.</summary>
    public int SampleRate { get; }
    /// <summary>Channel count; one is mono, two preserves separated-party stereo.</summary>
    public int Channels { get; }
    /// <summary>Number of audio frames, independent of channel count.</summary>
    public int FrameCount => Samples.Length / Channels;
    /// <summary>Audio duration computed from frames and sample rate.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds((double)FrameCount / SampleRate);
}
