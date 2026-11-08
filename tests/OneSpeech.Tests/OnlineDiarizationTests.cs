using Xunit;
using OneSpeech.Live;
using OneSpeech.Speakers;

namespace OneSpeech.Tests;

public sealed class OnlineDiarizationTests
{
    [Fact]
    public void BoundedCentroidsNeverMergeAndAssignWithoutAllocation()
    {
        var clusterer = new OnlineSpeakerClusterer(2, 2, 0.8);
        float[] a = [1, 0], b = [0, 1], c = [-1, 0];
        var first = clusterer.Assign(a);
        var second = clusterer.Assign(b);
        Assert.NotEqual(first, second);
        Assert.Equal(SpeakerLabel.Unassigned, clusterer.Assign(c));
        for (int i = 0; i < 100; i++) clusterer.Assign(a);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) clusterer.Assign(a);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(first, clusterer.Assign(a));
        Assert.Equal(2, clusterer.Count);
        Assert.Throws<ArgumentException>(() => clusterer.Assign([float.NaN, 0]));
        clusterer.Clear();
        Assert.Equal(0, clusterer.Count);
    }

    [Fact]
    public void FinalsKeepSpeakersAndTurnsUseOneClock()
    {
        using var inner = new Scripted();
        using var embedder = new AnalyticEmbedder();
        using var session = new OnlineDiarizationSession(inner, embedder);
        float[] a = new float[16000], b = new float[16000];
        Array.Fill(a, 1); Array.Fill(b, -1);
        var first = session.Push(new AudioBuffer(a, 16000), TestContext.Current.CancellationToken);
        var final = Assert.Single(first.OfType<FinalEvent>());
        var label = final.Segment.Words[0].Speaker;
        Assert.NotNull(label);
        Assert.Equal(SpeakerLabel.Unassigned, final.Segment.Words[1].Speaker);
        var turn = Assert.Single(first.OfType<SpeakerTurnEvent>());
        Assert.Equal(label, turn.Turn.Speaker);
        Assert.Same(turn.Turn, Assert.Single(final.SpeakerTurns));
        Assert.Equal(TimeSpan.Zero, turn.Turn.Start);
        Assert.Equal(TimeSpan.FromSeconds(1), turn.Turn.End);
        Assert.All(first, e => Assert.Equal(TimeSpan.FromSeconds(1), e.AudioTime));
        var second = session.Push(new AudioBuffer(b, 16000), TestContext.Current.CancellationToken);
        Assert.NotEqual(label, Assert.Single(second.OfType<FinalEvent>()).Segment.Words[0].Speaker);
        var third = session.Push(new AudioBuffer(a, 16000), TestContext.Current.CancellationToken);
        Assert.Equal(label, Assert.Single(third.OfType<FinalEvent>()).Segment.Words[0].Speaker);
        Assert.Equal(label, final.Segment.Words[0].Speaker);
        Assert.Equal(2, Assert.Single(third.OfType<FinalEvent>()).UtteranceId);
        Assert.Empty(session.Complete(TestContext.Current.CancellationToken));
        session.Dispose();
        Assert.False(embedder.Disposed);
        Assert.Empty(inner.Complete(TestContext.Current.CancellationToken));
    }

    [Fact]
    public void NonEmittingChunksAddNoWorkingAllocations()
    {
        using var inner = new SilentSession();
        using var embedder = new AnalyticEmbedder();
        using var session = new OnlineDiarizationSession(inner, embedder);
        var block = new AudioBuffer(new float[1600], 16000);
        var cancellationToken = TestContext.Current.CancellationToken;
        long allocated = -1;
        Exception? failure = null;
        // Measure only warmed library calls, on a thread without the runner's
        // execution context. Thread creation and all setup stay outside the counter.
        var worker = new Thread(() =>
        {
            try
            {
                for (int i = 0; i < 1000; i++) session.Push(block, cancellationToken);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 1000; i++) session.Push(block, cancellationToken);
                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            }
            catch (Exception exception) { failure = exception; }
        });
        using (ExecutionContext.SuppressFlow()) worker.Start();
        worker.Join();
        Assert.Null(failure);
        Assert.Equal(0, allocated);
        Assert.Equal(TimeSpan.FromSeconds(200), session.AudioTime);
    }

    [Fact]
    public void ChineseUsesIndependentEmbeddingSpaceAndIds()
    {
        using var inner = new Scripted("zh-CN");
        using var english = new AnalyticEmbedder();
        using var chinese = new AnalyticEmbedder();
        using var session = new OnlineDiarizationSession(inner, english, chineseEmbedder: chinese);
        var final = Assert.Single(session.Push(new AudioBuffer(new float[16000], 16000), TestContext.Current.CancellationToken).OfType<FinalEvent>());
        Assert.Equal("zh-speaker-1", final.Segment.Words[0].Speaker?.Id);
    }

    private sealed class SilentSession : LiveSession
    {
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken) { }
        protected override void Finish(CancellationToken cancellationToken) { }
    }

    private sealed class Scripted(string language = "en") : LiveSession
    {
        private TimeSpan start;
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
        {
            PublishPartial(new Segment { Text = "provisional", Start = start, End = AudioTime });
            PublishFinal(new Segment { Text = "speech", Language = language, Start = start, End = AudioTime,
                Words = [new Word { Text = "speech", Start = start, End = AudioTime }, new Word { Text = "untimed" }] });
            PublishEndpoint(AudioTime, EndpointReason.Silence);
            start = AudioTime;
        }
        protected override void Finish(CancellationToken cancellationToken) { }
    }

    // Analytic basis vectors exercise assignment, not a person's biometric embedding.
    private sealed class AnalyticEmbedder : SpeakerEmbedder
    {
        public bool Disposed { get; private set; }
        public override int Dimension => 2;
        public override int SampleRate => 16000;
        public override ModelInfo Model => throw new NotSupportedException();
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        { destination[0] = audio.Samples.Span[0] > 0 ? 1 : 0; destination[1] = audio.Samples.Span[0] > 0 ? 0 : 1; }
        public override void Dispose() => Disposed = true;
    }
}
