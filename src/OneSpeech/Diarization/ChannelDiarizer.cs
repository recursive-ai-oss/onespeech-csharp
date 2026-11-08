using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>Diarizes separated-party stereo by channel, not by voice identity. Borrows the detector for its lifetime.</summary>
public sealed class ChannelDiarizer : BlockDiarizer
{
    private readonly VoiceActivityDetector detector;
    private readonly bool forceSeparated;

    /// <summary>Creates a channel diarizer. By default whole-recording auto-detection must pass; forceSeparated bypasses that check, not the stereo requirement.
    /// Block processing requires 16 kHz input; the borrowed detector must accept mono 16 kHz. Throws ArgumentNullException for a null detector.</summary>
    public ChannelDiarizer(VoiceActivityDetector detector, bool forceSeparated = false)
    { ArgumentNullException.ThrowIfNull(detector); this.detector = detector; this.forceSeparated = forceSeparated; }

    /// <summary>Two channel speakers; simultaneous channel activity is retained as overlapping turns.</summary>
    public override Capabilities Capabilities { get; } = new() { MaxSpeakers = 2 };
    /// <summary>VAD model provenance; channel assignment itself has no model weights.</summary>
    public override ImmutableArray<ModelInfo> Models => [detector.Model];

    /// <summary>Tests stereo for separated-party energy: at least three 20 ms frames dominated by each channel by ≥20 dB,
    /// and absolute normalized waveform dot product ≤0.3. Silence and one-sided calls are inconclusive (false).
    /// Scans without retaining PCM; throws ArgumentException for nonfinite PCM and checks cancellation per frame.</summary>
    public static bool IsSeparated(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Channels != 2) return false;
        var stats = new SeparationStatistics(); stats.Add(audio, cancellationToken); return stats.Separated;
    }

    /// <summary>Runs VAD separately on channels in consecutive 16 kHz stereo blocks, returning globally consistent channel-1/channel-2 turns.
    /// Auto-detection is assessed across the whole recording; failure throws CapabilityNotSupportedException, never a downmix fallback.
    /// Temporary PCM is bounded to 30 seconds; output grows only with turns. VAD resets each 30 seconds. Cancels per block/frame.</summary>
    public override ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var stats = new SeparationStatistics(); var perChannel = new[] { new List<Turn>(), new List<Turn>() };
        long frames = 0;
        foreach (var audio in DiarizationBlocks.Fixed(blocks, cancellationToken))
        {
            if (audio.Channels != 2) throw new CapabilityNotSupportedException("Channel diarization requires two separated channels. Use a clustering diarizer for mixed audio.");
            stats.Add(audio, cancellationToken);
            for (int channel = 0; channel < 2; channel++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var mono = new float[audio.FrameCount];
                for (int i = 0; i < mono.Length; i++) mono[i] = audio.Samples.Span[i * 2 + channel];
                var origin = TimeSpan.FromSeconds((double)frames / 16000);
                var speaker = new SpeakerLabel { Id = "channel-" + (channel + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) };
                foreach (var interval in detector.Detect(new AudioBuffer(mono, 16000), cancellationToken).OrderBy(i => i.Start))
                {
                    if (interval.Start < TimeSpan.Zero || interval.End > audio.Duration || interval.End <= interval.Start)
                        throw new InvalidOperationException("VAD returned an interval outside its input block.");
                    var turn = new Turn { Speaker = speaker, Start = origin + interval.Start, End = origin + interval.End, Text = "" };
                    var turns = perChannel[channel];
                    if (turns.Count != 0 && turns[^1].End >= turn.Start) turns[^1] = turns[^1] with { End = TimeSpan.FromTicks(Math.Max(turn.End.Ticks, turns[^1].End.Ticks)) };
                    else turns.Add(turn);
                }
            }
            frames += audio.FrameCount;
        }
        cancellationToken.ThrowIfCancellationRequested();
        if (frames != 0 && !forceSeparated && !stats.Separated)
            throw new CapabilityNotSupportedException("Stereo separation is inconclusive. Use a clustering diarizer, or forceSeparated: true for known separated-party stereo.");
        return [.. perChannel.SelectMany(t => t).OrderBy(t => t.Start).ThenBy(t => t.Speaker.Id, StringComparer.Ordinal)];
    }

    private sealed class SeparationStatistics
    {
        private int left, right;
        private double aa, bb, ab;
        internal bool Separated => left >= 3 && right >= 3 && aa > 0 && bb > 0 && Math.Abs(ab / Math.Sqrt(aa * bb)) <= 0.3;
        internal void Add(AudioBuffer audio, CancellationToken token)
        {
            var samples = audio.Samples.Span;
            int size = Math.Max(1, audio.SampleRate / 50);
            for (int frame = 0; frame < audio.FrameCount; frame += size)
            {
                token.ThrowIfCancellationRequested(); double a = 0, b = 0;
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
        }
    }
}
