using System.Collections.Immutable;

namespace OneSpeech.Vad;

/// <summary>Silero segmentation parameters, in audio time. Defaults match the published timestamp API for unlimited speech length.</summary>
public sealed record SileroVadOptions
{
    /// <summary>Activity profile for offline speaker clustering, checked on AMI development audio: activation 0.2,
    /// deactivation 0.05, minimum speech/silence 250 ms, padding 150 ms. Recognition keeps the constructor defaults.</summary>
    public static SileroVadOptions Diarization => new()
    {
        Threshold = 0.2f, NegativeThreshold = 0.2f - 0.15f,
        MinimumSpeech = TimeSpan.FromMilliseconds(250), MinimumSilence = TimeSpan.FromMilliseconds(250),
        SpeechPadding = TimeSpan.FromMilliseconds(150)
    };

    /// <summary>Speech activation probability, default 0.5, finite in (0,1].</summary>
    public float Threshold { get; init; } = 0.5f;
    /// <summary>Speech deactivation probability, default 0.35; finite in [0,Threshold).</summary>
    public float NegativeThreshold { get; init; } = 0.35f;
    /// <summary>Minimum retained speech duration, default 250 ms; from zero through one hour.</summary>
    public TimeSpan MinimumSpeech { get; init; } = TimeSpan.FromMilliseconds(250);
    /// <summary>Silence required to close an interval, default 100 ms; from zero through one hour.</summary>
    public TimeSpan MinimumSilence { get; init; } = TimeSpan.FromMilliseconds(100);
    /// <summary>Padding on both sides of retained speech, default 30 ms; from zero through one second. Nearby pads share the gap.</summary>
    public TimeSpan SpeechPadding { get; init; } = TimeSpan.FromMilliseconds(30);

    internal void Validate()
    {
        if (!float.IsFinite(Threshold) || Threshold <= 0 || Threshold > 1) throw new ArgumentOutOfRangeException(nameof(Threshold));
        if (!float.IsFinite(NegativeThreshold) || NegativeThreshold < 0 || NegativeThreshold >= Threshold) throw new ArgumentOutOfRangeException(nameof(NegativeThreshold));
        if (MinimumSpeech < TimeSpan.Zero || MinimumSpeech > TimeSpan.FromHours(1)) throw new ArgumentOutOfRangeException(nameof(MinimumSpeech));
        if (MinimumSilence < TimeSpan.Zero || MinimumSilence > TimeSpan.FromHours(1)) throw new ArgumentOutOfRangeException(nameof(MinimumSilence));
        if (SpeechPadding < TimeSpan.Zero || SpeechPadding > TimeSpan.FromSeconds(1)) throw new ArgumentOutOfRangeException(nameof(SpeechPadding));
    }
}

// Frame positions are integers to avoid cumulative floating-point time drift.
internal sealed class SpeechSegmenter(int sampleRate, SileroVadOptions options)
{
    private readonly int frameSamples = sampleRate / 1000 * 32;
    private long position;
    private long start = -1;
    private long silence = -1;

    internal SpeechInterval? Push(float probability)
    {
        long current = position;
        position += frameSamples;
        if (probability >= options.Threshold)
        {
            silence = -1;
            if (start < 0) start = current;
        }
        else if (probability < options.NegativeThreshold && start >= 0)
        {
            if (silence < 0) silence = current;
            if (current - silence >= options.MinimumSilence.TotalSeconds * sampleRate)
            {
                long end = silence;
                long begin = start;
                start = silence = -1;
                if (end - begin > options.MinimumSpeech.TotalSeconds * sampleRate) return Interval(begin, end);
            }
        }
        return null;
    }

    internal SpeechInterval? Finish(long length)
    {
        if (start < 0 || length - start <= options.MinimumSpeech.TotalSeconds * sampleRate) return null;
        var result = Interval(start, length);
        start = silence = -1;
        return result;
    }

    private SpeechInterval Interval(long begin, long end) => new()
    {
        Start = TimeSpan.FromSeconds((double)begin / sampleRate), End = TimeSpan.FromSeconds((double)end / sampleRate)
    };

    internal static ImmutableArray<SpeechInterval> Pad(List<SpeechInterval> raw, TimeSpan length, TimeSpan padding)
    {
        var result = ImmutableArray.CreateBuilder<SpeechInterval>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            var left = i == 0 ? TimeSpan.Zero : raw[i - 1].End + (raw[i].Start - raw[i - 1].End) / 2;
            var right = i + 1 == raw.Count ? length : raw[i].End + (raw[i + 1].Start - raw[i].End) / 2;
            result.Add(new SpeechInterval { Start = Max(left, raw[i].Start - padding), End = Min(right, raw[i].End + padding) });
        }
        return result.MoveToImmutable();
    }

    private static TimeSpan Min(TimeSpan a, TimeSpan b) => a < b ? a : b;
    private static TimeSpan Max(TimeSpan a, TimeSpan b) => a > b ? a : b;
}
