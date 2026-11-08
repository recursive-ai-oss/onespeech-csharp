namespace OneSpeech.Audio;

/// <summary>A bounded mono 16 kHz recognition chunk with its origin on the input audio clock.</summary>
public sealed record AudioChunk
{
    /// <summary>Start relative to the original recording, including any retained padding.</summary>
    public required TimeSpan Start { get; init; }
    /// <summary>Owned mono PCM, no longer than the chunker's maximum. Safe to retain after advancing enumeration.</summary>
    public required AudioBuffer Audio { get; init; }
}

/// <summary>VAD-first bounded-horizon chunking. Speech regions are merged within the maximum duration; the last detected silence is preferred over cutting speech.</summary>
public sealed class VadChunker
{
    private readonly VoiceActivityDetector detector;
    private readonly int maximumSamples;
    private const int LookaheadSamples = 16000;

    /// <summary>Creates a chunker. Maximum defaults to 30 s and must be 1–3600 s. Analysis uses one second each of lookbehind and lookahead; detector calls reset per analysis horizon.</summary>
    public VadChunker(VoiceActivityDetector detector, TimeSpan? maximumDuration = null)
    {
        ArgumentNullException.ThrowIfNull(detector);
        this.detector = detector;
        var duration = maximumDuration ?? TimeSpan.FromSeconds(30);
        if (duration < TimeSpan.FromSeconds(1) || duration > TimeSpan.FromHours(1)) throw new ArgumentOutOfRangeException(nameof(maximumDuration));
        maximumSamples = checked((int)(duration.TotalSeconds * 16000));
    }

    /// <summary>Enumerates chunks from 16 kHz blocks, preserving audio gaps on their original clock. Channels are averaged, not concatenated. Input blocks may borrow storage until the next MoveNext. Reports processed audio time, including silence; cancellation is checked per input block and chunk.</summary>
    public IEnumerable<AudioChunk> Read(IEnumerable<AudioBuffer> blocks, IProgress<TimeSpan>? progress, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var pending = new float[maximumSamples + LookaheadSamples];
        var history = new float[LookaheadSamples];
        var analysis = new float[maximumSamples + 2 * LookaheadSamples];
        int count = 0, historyCount = 0;
        long origin = 0;
        foreach (var block in blocks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (block.SampleRate != 16000) throw new ArgumentException("Chunk input must be 16000 Hz; resample in-memory PCM with OneAudio first.");
            int position = 0;
            while (position < block.FrameCount)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int take = Math.Min(pending.Length - count, block.FrameCount - position);
                Mix(block, position, pending.AsSpan(count, take));
                count += take;
                position += take;
                if (count != pending.Length) continue;
                var (chunk, consumed) = Analyze(pending, count, history, historyCount, analysis, origin, false, cancellationToken);
                if (chunk is not null) yield return chunk;
                historyCount = RetainHistory(history, historyCount, pending, consumed);
                count -= consumed;
                pending.AsSpan(consumed, count).CopyTo(pending);
                origin += consumed;
                progress?.Report(TimeSpan.FromSeconds(origin / 16000d));
            }
        }
        while (count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (chunk, consumed) = Analyze(pending, count, history, historyCount, analysis, origin, true, cancellationToken);
            if (chunk is not null) yield return chunk;
            historyCount = RetainHistory(history, historyCount, pending, consumed);
            count -= consumed;
            pending.AsSpan(consumed, count).CopyTo(pending);
            origin += consumed;
            progress?.Report(TimeSpan.FromSeconds(origin / 16000d));
        }
        cancellationToken.ThrowIfCancellationRequested();
    }

    private (AudioChunk? Chunk, int Consumed) Analyze(float[] pending, int count, float[] history, int historyCount, float[] analysis, long origin, bool final, CancellationToken cancellationToken)
    {
        int limit = Math.Min(maximumSamples, count);
        history.AsSpan(0, historyCount).CopyTo(analysis);
        pending.AsSpan(0, count).CopyTo(analysis.AsSpan(historyCount));
        var regions = detector.Detect(new AudioBuffer(analysis.AsMemory(0, historyCount + count), 16000), cancellationToken);
        int first = -1, last = -1;
        foreach (var region in regions)
        {
            int rawEnd = (int)Math.Round(region.End.TotalSeconds * 16000) - historyCount;
            if (rawEnd <= 0) continue;
            int start = Math.Clamp((int)Math.Round(region.Start.TotalSeconds * 16000) - historyCount, 0, count);
            int end = Math.Clamp(rawEnd, start, count);
            if (start >= limit) break;
            if (first < 0) first = start;
            if (end <= limit) last = end;
            else if (last < 0) last = limit;
        }
        if (first < 0 || last <= first) return (null, limit);
        int consume = final && count <= maximumSamples ? count : last;
        return (new AudioChunk
        {
            Start = TimeSpan.FromSeconds((origin + first) / 16000d),
            Audio = new AudioBuffer(pending.AsMemory(first, last - first).ToArray(), 16000)
        }, consume);
    }

    private static int RetainHistory(float[] history, int historyCount, float[] pending, int consumed)
    {
        int fromPending = Math.Min(history.Length, consumed);
        int fromHistory = Math.Min(historyCount, history.Length - fromPending);
        history.AsSpan(historyCount - fromHistory, fromHistory).CopyTo(history);
        pending.AsSpan(consumed - fromPending, fromPending).CopyTo(history.AsSpan(fromHistory));
        return fromHistory + fromPending;
    }

    private static void Mix(AudioBuffer block, int offset, Span<float> destination)
    {
        var samples = block.Samples.Span;
        for (int frame = 0; frame < destination.Length; frame++)
        {
            double sum = 0;
            for (int channel = 0; channel < block.Channels; channel++) sum += samples[(offset + frame) * block.Channels + channel];
            destination[frame] = (float)(sum / block.Channels);
        }
    }

    internal static IEnumerable<AudioBuffer> Blocks(AudioFileSource source, CancellationToken cancellationToken)
    {
        var storage = new float[4096 * source.Channels];
        int frames;
        while ((frames = source.Read(storage, cancellationToken)) > 0)
            yield return new AudioBuffer(storage.AsMemory(0, frames * source.Channels), 16000, source.Channels);
    }
}
