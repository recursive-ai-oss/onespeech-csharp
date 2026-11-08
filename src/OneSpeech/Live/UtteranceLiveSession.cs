using OneSpeech.Vad;

namespace OneSpeech.Live;

/// <summary>VAD-endpointed, finals-only recognition for offline families. Borrows the recognizer and Silero model, owns independent recurrent VAD state. PCM memory is bounded by the maximum utterance length plus padding and one frame.</summary>
public sealed class UtteranceLiveSession : LiveSession
{
    private readonly SpeechRecognizer recognizer;
    private readonly SileroVadState? state;
    private readonly Func<ReadOnlyMemory<float>, CancellationToken, float>? probability;
    private readonly SileroVadOptions options;
    private readonly int maximumSamples;
    private readonly int padding;
    private readonly float[] ring;
    private readonly float[] frame = new float[512];
    private readonly float[] utterance;
    private string? language;
    private int frameCount;
    private long position;
    private long speechStart = -1;
    private long paddedStart;
    private long silenceStart = -1;
    private long lastEnd;

    /// <summary>Creates a finals-only session. Input is mono 16 kHz. Maximum utterances default to 30 s (range 1–300 s, rounded up to the next 32 ms VAD frame at closure). Options default to Silero's threshold 0.5, negative threshold 0.35, 250 ms minimum speech, 100 ms silence, 30 ms padding. Language null uses model detection. Models must outlive the session; cancellation is checked every frame and recognition call.</summary>
    public UtteranceLiveSession(SpeechRecognizer recognizer, SileroVad detector, string? language = null, TimeSpan? maximumUtteranceDuration = null, SileroVadOptions? options = null)
        : this(recognizer, language, maximumUtteranceDuration, options)
    {
        ArgumentNullException.ThrowIfNull(detector);
        state = detector.CreateState(16000);
    }

    internal UtteranceLiveSession(SpeechRecognizer recognizer, Func<ReadOnlyMemory<float>, CancellationToken, float> probability, string? language = null, TimeSpan? maximumUtteranceDuration = null, SileroVadOptions? options = null)
        : this(recognizer, language, maximumUtteranceDuration, options) => this.probability = probability;

    private UtteranceLiveSession(SpeechRecognizer recognizer, string? language, TimeSpan? maximumUtteranceDuration, SileroVadOptions? options)
    {
        ArgumentNullException.ThrowIfNull(recognizer);
        this.recognizer = recognizer;
        this.language = language;
        this.options = options ?? new();
        this.options.Validate();
        var maximum = maximumUtteranceDuration ?? TimeSpan.FromSeconds(30);
        if (maximum < TimeSpan.FromSeconds(1) || maximum > TimeSpan.FromSeconds(300)) throw new ArgumentOutOfRangeException(nameof(maximumUtteranceDuration));
        maximumSamples = checked((int)Math.Ceiling(maximum.TotalSeconds * 16000));
        padding = checked((int)Math.Round(this.options.SpeechPadding.TotalSeconds * 16000));
        ring = new float[maximumSamples + padding + frame.Length * 2];
        utterance = new float[ring.Length];
    }

    /// <inheritdoc />
    protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
    {
        var samples = audio.Samples.Span;
        while (!samples.IsEmpty)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(frame.Length - frameCount, samples.Length);
            samples[..count].CopyTo(frame.AsSpan(frameCount));
            for (int i = 0; i < count; i++) ring[(int)((position + i) % ring.Length)] = samples[i];
            frameCount += count;
            position += count;
            samples = samples[count..];
            if (frameCount == frame.Length)
            {
                Analyze(frame.Length, cancellationToken);
                frameCount = 0;
            }
        }
    }

    private void Analyze(int realSamples, CancellationToken cancellationToken)
    {
        long begin = position - realSamples;
        float p = state is not null ? state.ProcessFrame(frame, cancellationToken) : probability!(frame, cancellationToken);
        if (p >= options.Threshold)
        {
            silenceStart = -1;
            if (speechStart < 0)
            {
                speechStart = begin;
                paddedStart = Math.Max(lastEnd, Math.Max(0, begin - padding));
            }
        }
        else if (p < options.NegativeThreshold && speechStart >= 0)
        {
            if (silenceStart < 0) silenceStart = begin;
            if (position - realSamples - silenceStart >= options.MinimumSilence.TotalSeconds * 16000)
            {
                Close(Math.Min(position, silenceStart + padding), silenceStart, EndpointReason.Silence, cancellationToken);
                return;
            }
        }
        if (speechStart >= 0 && position - paddedStart >= maximumSamples)
            Close(position, silenceStart >= 0 ? silenceStart : position, EndpointReason.DurationLimit, cancellationToken);
    }

    /// <inheritdoc />
    protected override void Finish(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (frameCount != 0)
        {
            frame.AsSpan(frameCount).Clear();
            Analyze(frameCount, cancellationToken);
            frameCount = 0;
        }
        if (speechStart >= 0)
            Close(position, silenceStart >= 0 ? silenceStart : position, EndpointReason.EndOfInput, cancellationToken);
    }

    private void Close(long end, long speechEnd, EndpointReason reason, CancellationToken cancellationToken)
    {
        if (speechEnd - speechStart > options.MinimumSpeech.TotalSeconds * 16000)
        {
            int count = checked((int)(end - paddedStart));
            for (int i = 0; i < count; i++) utterance[i] = ring[(int)((paddedStart + i) % ring.Length)];
            var result = recognizer.Transcribe(new AudioBuffer(utterance.AsMemory(0, count), 16000), language, cancellationToken);
            language ??= result.Language;
            var offset = TimeSpan.FromSeconds(paddedStart / 16000d);
            if (result.Text.Length != 0)
                PublishFinal(new Segment
                {
                    Text = result.Text, Language = result.Language,
                    Start = offset, End = TimeSpan.FromSeconds(end / 16000d),
                    Words = [.. result.Words.Select(w => w with { Start = w.Start + offset, End = w.End + offset })]
                });
            PublishEndpoint(TimeSpan.FromSeconds(end / 16000d), reason);
        }
        lastEnd = end;
        speechStart = silenceStart = -1;
    }

    /// <inheritdoc />
    protected override void DisposeCore() => state?.Dispose();
}
