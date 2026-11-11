using System.Collections.Immutable;
using OneSpeech.Speakers;

namespace OneSpeech.Live;

/// <summary>Adds causal, non-overlap-aware diarization to a borrowed live recognizer. Turns and word labels are committed at utterance closure; earlier finals are never revised. No retrospective merging. Not thread safe.</summary>
public sealed class OnlineDiarizationSession : LiveSession
{
    private readonly LiveSession inner;
    private readonly SpeakerEmbedder embedder;
    private readonly SpeakerEmbedder? chineseEmbedder;
    private readonly OnlineSpeakerClusterer clusters;
    private readonly OnlineSpeakerClusterer? chineseClusters;
    private readonly float[] vector;
    private readonly float[]? chineseVector;
    private readonly float[] ring = new float[320 * 16000];
    private readonly float[] window = new float[24000];
    private long position;

    /// <summary>Creates a decorator borrowing all resources. Embedders must accept mono 16000 Hz. Windows are at most 1.5 s, without lookahead beyond the endpoint. Capacity defaults to 16 per embedder, threshold 0.3. Optional Chinese embedder is selected for zh/Chinese segment languages; distinct ID prefixes avoid cross-language vector comparisons. Bounds/errors follow OnlineSpeakerClusterer. Retains 320 s PCM; pushes over 10 s or finals longer than 300 s are refused. Dispose clears centroids/audio, not borrowed resources.</summary>
    public OnlineDiarizationSession(LiveSession inner, SpeakerEmbedder embedder, double minimumSimilarity = 0.3, int maximumSpeakers = 16, SpeakerEmbedder? chineseEmbedder = null)
    {
        ArgumentNullException.ThrowIfNull(inner);
        ArgumentNullException.ThrowIfNull(embedder);
        if (inner.AudioTime != TimeSpan.Zero) throw new ArgumentException("Wrap a fresh session before pushing audio.", nameof(inner));
        if (embedder.SampleRate != 16000 || chineseEmbedder is { SampleRate: not 16000 }) throw new ArgumentException("Live embedders require 16000 Hz.", nameof(embedder));
        this.inner = inner; this.embedder = embedder; this.chineseEmbedder = chineseEmbedder;
        clusters = new(embedder.Dimension, maximumSpeakers, minimumSimilarity);
        vector = new float[embedder.Dimension];
        if (chineseEmbedder is not null)
        {
            chineseClusters = new(chineseEmbedder.Dimension, maximumSpeakers, minimumSimilarity, "zh-speaker");
            chineseVector = new float[chineseEmbedder.Dimension];
        }
    }

    /// <inheritdoc />
    protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
    {
        if (audio.FrameCount > 160000) throw new ArgumentException("Online diarization accepts chunks of at most 10 seconds.");
        foreach (float sample in audio.Samples.Span) ring[(int)(position++ % ring.Length)] = sample;
        Forward(inner.Push(audio, cancellationToken), cancellationToken);
    }

    /// <inheritdoc />
    protected override void Finish(CancellationToken cancellationToken) => Forward(inner.Complete(cancellationToken), cancellationToken);

    private void Forward(ImmutableArray<LiveEvent> events, CancellationToken cancellationToken)
    {
        foreach (var update in events)
        {
            switch (update)
            {
                case PartialEvent partial: PublishPartial(partial.Segment); break;
                case FinalEvent final:
                    if (!final.SpeakerTurns.IsDefaultOrEmpty) throw new CapabilityNotSupportedException("Wrap a recognition-only session, not an already diarized session.");
                    Commit(final.Segment, cancellationToken); break;
                case EndpointEvent endpoint: PublishEndpoint(endpoint.End, endpoint.Reason); break;
                // Existing diarization must not be silently relabelled.
                case SpeakerTurnEvent: throw new CapabilityNotSupportedException("Wrap a recognition-only session, not an already diarized session.");
            }
        }
    }

    private void Commit(Segment segment, CancellationToken cancellationToken)
    {
        if (segment.Start is not { } start || segment.End is not { } end) throw new CapabilityNotSupportedException("Online diarization requires segment boundaries.");
        long begin = (long)Math.Round(start.TotalSeconds * 16000), stop = (long)Math.Round(end.TotalSeconds * 16000);
        if (begin < Math.Max(0, position - ring.Length) || stop > position || stop < begin || stop - begin > 300 * 16000)
            throw new CapabilityNotSupportedException("Online diarization requires retained utterances of at most 300 seconds.");
        bool chinese = chineseEmbedder is not null && segment.Language is { } language &&
            (language.Equals("Chinese", StringComparison.OrdinalIgnoreCase) || language.Equals("zh", StringComparison.OrdinalIgnoreCase) || language.StartsWith("zh-", StringComparison.OrdinalIgnoreCase));
        var model = chinese ? chineseEmbedder! : embedder;
        var clustering = chinese ? chineseClusters! : clusters;
        var embedding = chinese ? chineseVector! : vector;
        var turns = new List<Turn>();
        for (long at = begin; at < stop;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int length = (int)Math.Min(window.Length, stop - at);
            // Fold short tails into the preceding window rather than embedding <45 ms.
            if (stop - at > window.Length && stop - at - window.Length < 720) length = (int)((stop - at) / 2);
            SpeakerLabel label = SpeakerLabel.Unassigned;
            if (length >= 720)
            {
                for (int i = 0; i < length; i++) window[i] = ring[(int)((at + i) % ring.Length)];
                model.Embed(new AudioBuffer(window.AsMemory(0, length), 16000), embedding, cancellationToken);
                label = clustering.Assign(embedding);
            }
            // Windows of one utterance that land on the same speaker are one turn; merging happens here,
            // before anything is emitted, so emitted events stay immutable.
            var turnEnd = TimeSpan.FromSeconds((at + length) / 16000d);
            if (turns.Count > 0 && turns[^1].Speaker == label) turns[^1] = turns[^1] with { End = turnEnd };
            else turns.Add(new Turn { Start = TimeSpan.FromSeconds(at / 16000d), End = turnEnd, Speaker = label, Text = "" });
            at += length;
        }
        var words = ImmutableArray.CreateBuilder<Word>(segment.Words.Length);
        foreach (var word in segment.Words)
        {
            var speaker = SpeakerLabel.Unassigned;
            long overlap = 0;
            if (word.Start is { } ws && word.End is { } we)
                foreach (var turn in turns)
                {
                    long duration = Math.Min(we.Ticks, turn.End.Ticks) - Math.Max(ws.Ticks, turn.Start.Ticks);
                    if (duration > overlap) { overlap = duration; speaker = turn.Speaker; }
                }
            words.Add(word with { Speaker = speaker });
        }
        foreach (var turn in turns) PublishSpeakerTurn(turn);
        PublishFinal(segment with { Words = words.MoveToImmutable() }, turns.ToImmutableArray());
    }

    /// <inheritdoc />
    protected override void DisposeCore()
    {
        clusters.Clear(); chineseClusters?.Clear(); Array.Clear(ring); Array.Clear(window); Array.Clear(vector);
        if (chineseVector is not null) Array.Clear(chineseVector);
    }
}
