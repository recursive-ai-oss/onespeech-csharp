using System.Collections.Immutable;
using System.Runtime.CompilerServices;

namespace OneSpeech;

/// <summary>A single-use, non-thread-safe push recognizer accepting mono 16 kHz PCM. Push and Complete are synchronous CPU work; Listen awaits only its input. Dispose never completes pending speech.</summary>
public abstract class LiveSession : IDisposable
{
    private readonly List<LiveEvent> pending = [];
    private long frames;
    private long utterance;
    private long revision;
    private bool finalized;
    private bool completed;
    private bool disposed;
    private bool faulted;

    /// <summary>Accepted audio duration, computed from integer samples at 16000 Hz. Starts at zero.</summary>
    public TimeSpan AudioTime => TimeSpan.FromSeconds(frames / 16000d);

    /// <summary>Consumes borrowed PCM before returning immutable event snapshots. Empty input is a no-op. Throws UnsupportedAudioException for non-mono/non-16-kHz or nonfinite input, InvalidOperationException after completion/failure, ObjectDisposedException after disposal, or OperationCanceledException on cancellation. Input validation/cancellation before processing leaves the session usable; processing failures fault it.</summary>
    public ImmutableArray<LiveEvent> Push(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        EnsureOpen();
        cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new UnsupportedAudioException("Live sessions require mono PCM at 16000 Hz. Resample and downmix with OneAudio before pushing.");
        foreach (float sample in audio.Samples.Span)
            if (!float.IsFinite(sample)) throw new UnsupportedAudioException("Live sessions require finite PCM samples.");
        if (audio.FrameCount == 0) return [];
        frames = checked(frames + audio.FrameCount);
        try { Process(audio, cancellationToken); return Drain(); }
        catch { faulted = true; pending.Clear(); throw; }
    }

    /// <summary>Flushes pending speech without inventing samples. Repeated completion returns no events. Other lifecycle/cancellation errors follow Push.</summary>
    public ImmutableArray<LiveEvent> Complete(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (faulted) throw new InvalidOperationException("The live session failed; create a new session.");
        cancellationToken.ThrowIfCancellationRequested();
        if (completed) return [];
        try { Finish(cancellationToken); completed = true; return Drain(); }
        catch { faulted = true; pending.Clear(); throw; }
    }

    /// <summary>Awaits borrowed PCM blocks, yields their updates, and completes on normal end-of-input. Does not dispose this session or own the source. Early iterator disposal/cancellation does not flush pending speech. Do not interleave with Push or another Listen.</summary>
    public async IAsyncEnumerable<LiveEvent> Listen(IAsyncEnumerable<AudioBuffer> source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        EnsureOpen();
        await foreach (var block in source.WithCancellation(cancellationToken).ConfigureAwait(false))
            foreach (var update in Push(block, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return update;
            }
        foreach (var update in Complete(cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return update;
        }
    }

    /// <summary>Consumes a block. AudioTime includes this entire block; storage must not be retained unless copied.</summary>
    protected abstract void Process(AudioBuffer audio, CancellationToken cancellationToken);
    /// <summary>Flushes real input at end-of-input. Never append synthetic silence to the session clock.</summary>
    protected abstract void Finish(CancellationToken cancellationToken);
    /// <summary>Publishes a replacing hypothesis with an automatically increasing revision. Throws InvalidOperationException after a FinalEvent and before its EndpointEvent, or for timing outside accepted audio.</summary>
    protected void PublishPartial(Segment segment)
    {
        if (finalized) throw new InvalidOperationException("A FinalEvent cannot be revised.");
        Validate(segment);
        pending.Add(new PartialEvent { AudioTime = AudioTime, UtteranceId = utterance, Revision = checked(++revision), Segment = segment });
    }
    /// <summary>Commits one immutable hypothesis for the current utterance. Throws InvalidOperationException for a second FinalEvent before EndpointEvent, or for timing outside accepted audio.</summary>
    protected void PublishFinal(Segment segment) => PublishFinal(segment, []);

    /// <summary>Commits a final with immutable speaker activity, including for words without timing. Bounds/errors follow PublishFinal and PublishSpeakerTurn; default arrays are rejected with ArgumentException. Does not publish duplicate SpeakerTurnEvents automatically.</summary>
    protected void PublishFinal(Segment segment, ImmutableArray<Turn> speakerTurns)
    {
        if (finalized) throw new InvalidOperationException("An utterance already has a FinalEvent.");
        Validate(segment);
        if (speakerTurns.IsDefault) throw new ArgumentException("Speaker turns must be initialized.", nameof(speakerTurns));
        foreach (var turn in speakerTurns)
        {
            if (turn is null || turn.Start < TimeSpan.Zero || turn.End < turn.Start || turn.End > AudioTime) throw new ArgumentOutOfRangeException(nameof(speakerTurns));
        }
        pending.Add(new FinalEvent { AudioTime = AudioTime, UtteranceId = utterance, Segment = segment, SpeakerTurns = speakerTurns });
        finalized = true;
    }
    /// <summary>Closes this utterance and starts a new revision sequence. End must be between zero and AudioTime; throws ArgumentOutOfRangeException otherwise.</summary>
    protected void PublishEndpoint(TimeSpan end, EndpointReason reason)
    {
        if (end < TimeSpan.Zero || end > AudioTime) throw new ArgumentOutOfRangeException(nameof(end));
        pending.Add(new EndpointEvent { AudioTime = AudioTime, UtteranceId = utterance, End = end, Reason = reason });
        utterance = checked(utterance + 1);
        revision = 0;
        finalized = false;
    }
    /// <summary>Publishes speaker activity. Throws ArgumentOutOfRangeException for intervals outside accepted audio or with end before start.</summary>
    protected void PublishSpeakerTurn(Turn turn)
    {
        ArgumentNullException.ThrowIfNull(turn);
        if (turn.Start < TimeSpan.Zero || turn.End < turn.Start || turn.End > AudioTime) throw new ArgumentOutOfRangeException(nameof(turn));
        pending.Add(new SpeakerTurnEvent { AudioTime = AudioTime, UtteranceId = utterance, Turn = turn });
    }
    /// <summary>Releases implementation resources; invoked once. Borrowed models are not owned by the base class.</summary>
    protected virtual void DisposeCore() { }
    /// <summary>Discards pending speech and releases session resources. Idempotent.</summary>
    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        pending.Clear();
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    private void EnsureOpen()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (completed || faulted) throw new InvalidOperationException("The live session is complete or failed; create a new session.");
    }
    private ImmutableArray<LiveEvent> Drain()
    {
        if (pending.Count == 0) return [];
        var result = pending.ToImmutableArray();
        pending.Clear();
        return result;
    }
    private void Validate(Segment segment)
    {
        ArgumentNullException.ThrowIfNull(segment);
        Check(segment.Start, segment.End);
        foreach (var word in segment.Words) Check(word.Start, word.End);
    }
    private void Check(TimeSpan? start, TimeSpan? end)
    {
        if (start < TimeSpan.Zero || end < TimeSpan.Zero || start > AudioTime || end > AudioTime || end < start)
            throw new InvalidOperationException("Live output timing must lie on the accepted audio clock, with end at or after start.");
    }
}
