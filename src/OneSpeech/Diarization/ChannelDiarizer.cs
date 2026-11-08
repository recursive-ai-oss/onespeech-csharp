using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>Diarizes separated-party stereo by channel, not by voice identity. Borrows the detector for its lifetime.</summary>
public sealed class ChannelDiarizer : Diarizer
{
    private readonly VoiceActivityDetector detector;
    private readonly bool forceSeparated;

    /// <summary>Creates a channel diarizer. By default auto-detection must pass; forceSeparated bypasses that check, not the two-channel requirement.
    /// Detector must accept mono at the supplied rate. Throws ArgumentNullException for a null detector.</summary>
    public ChannelDiarizer(VoiceActivityDetector detector, bool forceSeparated = false)
    {
        ArgumentNullException.ThrowIfNull(detector);
        this.detector = detector;
        this.forceSeparated = forceSeparated;
    }

    /// <summary>Two channel speakers; simultaneous channel activity is retained as overlapping turns.</summary>
    public override Capabilities Capabilities { get; } = new() { MaxSpeakers = 2 };

    /// <summary>Tests stereo for separated-party energy: at least three 20 ms frames dominated by each channel by ≥20 dB,
    /// and absolute normalized waveform dot product ≤0.3. Silence and one-sided calls are inconclusive (false).
    /// Scans without retaining PCM; throws ArgumentException for nonfinite PCM and checks cancellation per frame.</summary>
    public static bool IsSeparated(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Channels != 2) return false;
        var samples = audio.Samples.Span;
        int size = Math.Max(1, audio.SampleRate / 50);
        int left = 0, right = 0;
        double aa = 0, bb = 0, ab = 0;
        for (int frame = 0; frame < audio.FrameCount; frame += size)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double a = 0, b = 0;
            for (int i = frame; i < Math.Min(audio.FrameCount, frame + size); i++)
            {
                double x = samples[2 * i], y = samples[2 * i + 1];
                if (!double.IsFinite(x) || !double.IsFinite(y)) throw new ArgumentException("PCM must be finite.", nameof(audio));
                a += x * x; b += y * y; ab += x * y;
            }
            aa += a; bb += b;
            if (a > 1e-8 * size && a >= 100 * b) left++;
            if (b > 1e-8 * size && b >= 100 * a) right++;
        }
        return left >= 3 && right >= 3 && aa > 0 && bb > 0 && Math.Abs(ab / Math.Sqrt(aa * bb)) <= 0.3;
    }

    /// <summary>Runs VAD separately on each channel and returns ordered turns named channel-1/channel-2.
    /// Auto-detection failure throws CapabilityNotSupportedException: use a voice-clustering diarizer or explicitly force separation.
    /// PCM is copied at most 30 seconds per channel at a time; cancellation is checked per block. No cross-block VAD-state promise is made.</summary>
    public override ImmutableArray<Turn> Diarize(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        if (audio.Channels != 2 || (!forceSeparated && !IsSeparated(audio, cancellationToken)))
            throw new CapabilityNotSupportedException("Channel diarization requires separated-party stereo. Use a clustering diarizer, or forceSeparated: true for known separated stereo.");
        var turns = new List<Turn>();
        int blockSize = (int)Math.Min(int.MaxValue, (long)audio.SampleRate * 30);
        for (int channel = 0; channel < 2; channel++)
        {
            var speaker = new SpeakerLabel { Id = "channel-" + (channel + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) };
            for (int offset = 0; offset < audio.FrameCount; offset += Math.Min(blockSize, audio.FrameCount - offset))
            {
                cancellationToken.ThrowIfCancellationRequested();
                int length = Math.Min(blockSize, audio.FrameCount - offset);
                var mono = new float[length];
                for (int i = 0; i < length; i++) mono[i] = audio.Samples.Span[(offset + i) * 2 + channel];
                var origin = TimeSpan.FromSeconds((double)offset / audio.SampleRate);
                foreach (var interval in detector.Detect(new AudioBuffer(mono, audio.SampleRate), cancellationToken))
                {
                    if (interval.Start < TimeSpan.Zero || interval.End > TimeSpan.FromSeconds((double)length / audio.SampleRate) || interval.End <= interval.Start)
                        throw new InvalidOperationException("VAD returned an interval outside its input block.");
                    turns.Add(new Turn { Speaker = speaker, Start = origin + interval.Start, End = origin + interval.End, Text = "" });
                }
            }
        }
        return [.. turns.OrderBy(t => t.Start).ThenBy(t => t.Speaker.Id, StringComparer.Ordinal)];
    }
}
