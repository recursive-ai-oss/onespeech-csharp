using System.Collections.Immutable;
using OneSpeech.Live;
using OneSpeech.Vad;
using Xunit;

namespace OneSpeech.Tests;

public sealed class LiveSessionTests
{
    private static readonly ModelInfo Info = new() { Id = "synthetic", Source = "synthetic", Revision = "1", Tier = ModelTier.Tier1, License = "MIT", Attribution = "" };
    private static readonly SileroVadOptions Options = new() { MinimumSpeech = TimeSpan.Zero, MinimumSilence = TimeSpan.FromMilliseconds(64), SpeechPadding = TimeSpan.Zero };

    [Fact]
    public async Task FileAndLiveFinalsMatchOnDecodedFixture()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "1089-134686-0001.flac");
        using var vad = new SileroVad(cancellationToken: TestContext.Current.CancellationToken);
        var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 }, cancellationToken: TestContext.Current.CancellationToken);
        using var source = OneSpeech.Audio.AudioFileSource.Open(path, TestContext.Current.CancellationToken);
        using var file = new UtteranceLiveSession(new FakeRecognizer(), vad);
        var expected = OneSpeech.Pipeline.SpeechServices.ReadFinals(source, file, TestContext.Current.CancellationToken).ToArray();
        using var live = new UtteranceLiveSession(new FakeRecognizer(), vad);
        var actual = new List<Segment>();
        await foreach (var update in live.Listen(ClipBlocks(clip.Samples), TestContext.Current.CancellationToken))
            if (update is FinalEvent final) actual.Add(final.Segment);
        Assert.NotEmpty(expected);
        Assert.Equal(expected.Select(s => (s.Text, s.Start, s.End)), actual.Select(s => (s.Text, s.Start, s.End)));
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].Words, actual[i].Words);
    }

    private static async IAsyncEnumerable<AudioBuffer> ClipBlocks(ReadOnlyMemory<float> samples)
    {
        for (int offset = 0; offset < samples.Length; offset += 137)
        {
            await Task.Yield();
            yield return new AudioBuffer(samples.Slice(offset, Math.Min(137, samples.Length - offset)), 16000);
        }
    }

    [Fact]
    public void UtteranceSessionIsIndependentOfPushBoundaries()
    {
        // Three speech frames, five silence frames, two speech frames, then a short tail.
        float[] samples = [.. Enumerable.Repeat(1f, 1536), .. new float[2560], .. Enumerable.Repeat(1f, 1024), .. Enumerable.Repeat(1f, 23)];
        var whole = Run(samples, samples.Length);
        var fragmented = Run(samples, 137);
        var a = whole.OfType<FinalEvent>().ToArray();
        var b = fragmented.OfType<FinalEvent>().ToArray();
        Assert.Equal(2, a.Length);
        Assert.Equal(a.Select(f => (f.UtteranceId, f.Segment.Text, f.Segment.Start, f.Segment.End)), b.Select(f => (f.UtteranceId, f.Segment.Text, f.Segment.Start, f.Segment.End)));
        Assert.Equal(TimeSpan.Zero, a[0].Segment.Start);
        Assert.Equal(TimeSpan.FromSeconds(1536 / 16000d), a[0].Segment.End);
        Assert.Equal(TimeSpan.FromSeconds(4096 / 16000d), a[1].Segment.Start);
        Assert.Equal(TimeSpan.FromSeconds(samples.Length / 16000d), a[1].Segment.End);
        Assert.All(fragmented.OfType<FinalEvent>(), f => Assert.Null(f.Segment.Words[1].Start));
        Assert.All(fragmented.OfType<FinalEvent>(), f => Assert.Equal(f.Segment.Start, f.Segment.Words[0].Start));
        Assert.Equal([EndpointReason.Silence, EndpointReason.EndOfInput], fragmented.OfType<EndpointEvent>().Select(e => e.Reason));
        Assert.Empty(fragmented.OfType<PartialEvent>());
        Assert.True(fragmented.Zip(fragmented.Skip(1)).All(pair => pair.First.AudioTime <= pair.Second.AudioTime));
        Assert.All(fragmented.OfType<FinalEvent>(), f => Assert.True(f.Segment.End <= f.AudioTime));
    }

    [Fact]
    public void LongSpeechSplitsWithoutDroppingOrDuplicatingSamples()
    {
        var recognizer = new FakeRecognizer();
        using var session = Create(recognizer, TimeSpan.FromSeconds(1));
        float[] samples = Enumerable.Repeat(1f, 512 * 100 + 17).ToArray();
        var events = new List<LiveEvent>();
        for (int i = 0; i < samples.Length; i += 777)
            events.AddRange(session.Push(new AudioBuffer(samples.AsMemory(i, Math.Min(777, samples.Length - i)), 16000), TestContext.Current.CancellationToken));
        events.AddRange(session.Complete(TestContext.Current.CancellationToken));
        Assert.Equal(samples.Length, recognizer.FrameCounts.Sum());
        Assert.All(recognizer.FrameCounts, n => Assert.InRange(n, 1, 16000 + 512));
        var finals = events.OfType<FinalEvent>().ToArray();
        Assert.Equal(4, finals.Length);
        for (int i = 1; i < finals.Length; i++) Assert.Equal(finals[i - 1].Segment.End, finals[i].Segment.Start);
        Assert.Equal(TimeSpan.FromSeconds(samples.Length / 16000d), session.AudioTime);
    }

    [Fact]
    public void SilenceEmptyAndCompletionAreWellDefined()
    {
        var recognizer = new FakeRecognizer();
        using var session = Create(recognizer);
        Assert.Empty(session.Push(new AudioBuffer(ReadOnlyMemory<float>.Empty, 16000), TestContext.Current.CancellationToken));
        Assert.Empty(session.Push(new AudioBuffer(new float[16013], 16000), TestContext.Current.CancellationToken));
        Assert.Empty(session.Complete(TestContext.Current.CancellationToken));
        Assert.Empty(session.Complete(TestContext.Current.CancellationToken));
        Assert.Empty(recognizer.FrameCounts);
        Assert.Throws<InvalidOperationException>(() => session.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.Complete(TestContext.Current.CancellationToken));
    }

    [Fact]
    public void FileStreamingIsLazyAndRefusesCancellationBeforeLoadingModels()
    {
        var segments = Speech.TranscribeStream("not-opened", new CancellationToken(true));
        Assert.Throws<OperationCanceledException>(() => segments.ToArray());
    }

    [Fact]
    public void ValidationAndEarlyCancellationDoNotAdvanceTheClock()
    {
        using var session = Create(new FakeRecognizer());
        Assert.Throws<UnsupportedAudioException>(() => session.Push(new AudioBuffer(new float[10], 8000), TestContext.Current.CancellationToken));
        Assert.Throws<UnsupportedAudioException>(() => session.Push(new AudioBuffer(new float[10], 16000, 2), TestContext.Current.CancellationToken));
        Assert.Throws<OperationCanceledException>(() => session.Push(new AudioBuffer(new float[10], 16000), new CancellationToken(true)));
        Assert.Equal(TimeSpan.Zero, session.AudioTime);
        Assert.Empty(session.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void RevisionsReplaceOnlyTheCurrentUtteranceAndFinalsStayImmutable()
    {
        using var session = new ScriptedSession();
        var block = new AudioBuffer(new float[512], 16000);
        var first = Assert.IsType<PartialEvent>(Assert.Single(session.Push(block, TestContext.Current.CancellationToken)));
        var second = Assert.IsType<PartialEvent>(Assert.Single(session.Push(block, TestContext.Current.CancellationToken)));
        Assert.Equal(1, first.Revision);
        Assert.Equal(2, second.Revision);
        var closing = session.Push(block, TestContext.Current.CancellationToken);
        var final = Assert.IsType<FinalEvent>(closing[0]);
        Assert.IsType<EndpointEvent>(closing[1]);
        var next = Assert.IsType<PartialEvent>(Assert.Single(session.Push(block, TestContext.Current.CancellationToken)));
        Assert.Equal(1, next.UtteranceId);
        Assert.Equal(1, next.Revision);
        Assert.Equal("final", final.Segment.Text);
        Assert.Equal(TimeSpan.FromSeconds(1536 / 16000d), final.AudioTime);
        Assert.Equal("partial-1", first.Segment.Text);
    }

    [Fact]
    public void ProcessingFailureFaultsSessionAndInvalidOutputIsRefused()
    {
        using var session = new BadSession();
        Assert.Throws<InvalidOperationException>(() => session.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => session.Complete(TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => session.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
        using var revising = new RevisingSession();
        Assert.Throws<InvalidOperationException>(() => revising.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task AsyncInputFlushesOnNormalEndButNotCancellation()
    {
        var recognizer = new FakeRecognizer();
        using var session = Create(recognizer);
        var events = new List<LiveEvent>();
        await foreach (var e in session.Listen(Blocks(), TestContext.Current.CancellationToken)) events.Add(e);
        Assert.Single(events.OfType<FinalEvent>());
        Assert.Equal(EndpointReason.EndOfInput, Assert.Single(events.OfType<EndpointEvent>()).Reason);
        Assert.Equal(1024, Assert.Single(recognizer.FrameCounts));
        using var canceled = Create(new FakeRecognizer());
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in canceled.Listen(Blocks(), new CancellationToken(true))) { }
        });
        Assert.Equal(TimeSpan.Zero, canceled.AudioTime);
    }

    private static async IAsyncEnumerable<AudioBuffer> Blocks()
    {
        yield return new AudioBuffer(Enumerable.Repeat(1f, 512).ToArray(), 16000);
        await Task.Yield();
        yield return new AudioBuffer(Enumerable.Repeat(1f, 512).ToArray(), 16000);
    }
    private static UtteranceLiveSession Create(FakeRecognizer recognizer, TimeSpan? maximum = null) =>
        new(recognizer, (frame, ct) => { ct.ThrowIfCancellationRequested(); return frame.Span[0] > 0 ? 1f : 0f; }, maximumUtteranceDuration: maximum, options: Options);
    private static List<LiveEvent> Run(float[] samples, int size)
    {
        using var session = Create(new FakeRecognizer());
        var result = new List<LiveEvent>();
        for (int i = 0; i < samples.Length; i += size)
            result.AddRange(session.Push(new AudioBuffer(samples.AsMemory(i, Math.Min(size, samples.Length - i)), 16000), TestContext.Current.CancellationToken));
        result.AddRange(session.Complete(TestContext.Current.CancellationToken));
        return result;
    }
    private sealed class FakeRecognizer : SpeechRecognizer
    {
        internal List<int> FrameCounts { get; } = [];
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { Languages = ["en"] };
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            FrameCounts.Add(audio.FrameCount);
            return new Transcript { Text = $"{audio.FrameCount}", Language = "en", Duration = audio.Duration,
                Words = [new Word { Text = "timed", Start = TimeSpan.Zero, End = audio.Duration, TimingMethod = TimingMethod.Native }, new Word { Text = "untimed" }] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => [];
    }
    private sealed class ScriptedSession : LiveSession
    {
        private int calls;
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
        {
            calls++;
            if (calls == 3)
            {
                PublishFinal(new Segment { Text = "final", Start = TimeSpan.Zero, End = AudioTime });
                PublishEndpoint(AudioTime, EndpointReason.Silence);
            }
            else PublishPartial(new Segment { Text = $"partial-{calls}" });
        }
        protected override void Finish(CancellationToken cancellationToken) { }
    }
    private sealed class BadSession : LiveSession
    {
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken) => PublishFinal(new Segment { Text = "bad", End = AudioTime + TimeSpan.FromSeconds(1) });
        protected override void Finish(CancellationToken cancellationToken) { }
    }
    private sealed class RevisingSession : LiveSession
    {
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
        {
            PublishFinal(new Segment { Text = "done" });
            PublishPartial(new Segment { Text = "illegal" });
        }
        protected override void Finish(CancellationToken cancellationToken) { }
    }
}
