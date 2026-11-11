using System.Collections.Immutable;
using OneSpeech.Pipeline;
using Xunit;

namespace OneSpeech.Tests;

public sealed class TimingOrderTests
{
    private static readonly ModelInfo Info = new() { Id = "constructed", Source = "synthetic", Revision = "1", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Constructed test." };

    [Fact]
    public void CtcPrecedesDtwButFailedForcedAlignmentFallsBack()
    {
        var first = new TestAligner(succeeds: false);
        var ctc = new TestAligner(succeeds: true);
        var pipeline = new VadTranscriptionPipeline(new TestRecognizer(native: false), new TestVad()) { TimingAligners = [first, ctc] };
        var result = pipeline.Transcribe(new AudioBuffer(new float[16000], 16000), "en", TestContext.Current.CancellationToken);
        Assert.Equal(1, first.Calls); Assert.Equal(1, ctc.Calls);
        Assert.Equal(TimingMethod.CtcAligner, result.Words[0].TimingMethod);
        Assert.Equal(TimeSpan.FromSeconds(0.2), result.Words[0].Start);
        Assert.Equal(result.Words[0], result.Segments[0].Words[0]);
        var fallback = new VadTranscriptionPipeline(new TestRecognizer(native: false), new TestVad()) { TimingAligners = [first] };
        Assert.Equal(TimingMethod.Dtw, fallback.Transcribe(new AudioBuffer(new float[16000], 16000), "en", TestContext.Current.CancellationToken).Words[0].TimingMethod);
    }

    [Fact]
    public void NativeAndUnsupportedLanguagesDoNotCallFallback()
    {
        var aligner = new TestAligner(succeeds: true);
        var native = new VadTranscriptionPipeline(new TestRecognizer(native: true), new TestVad()) { TimingAligners = [aligner] };
        Assert.Equal(TimingMethod.Native, native.Transcribe(new AudioBuffer(new float[16000], 16000), "en", TestContext.Current.CancellationToken).Words[0].TimingMethod);
        var unsupported = new VadTranscriptionPipeline(new TestRecognizer(native: false), new TestVad()) { TimingAligners = [aligner] };
        Assert.Equal(TimingMethod.Dtw, unsupported.Transcribe(new AudioBuffer(new float[16000], 16000), "fr", TestContext.Current.CancellationToken).Words[0].TimingMethod);
        Assert.Equal(0, aligner.Calls);
    }

    [Fact]
    public void TextOnlyAlignerUsesAdditiveTranscriptOverloadWithoutSchemaChanges()
    {
        Aligner aligner = new TextOnlyAligner();
        var original = new Word { Text = "HELLO,", Confidence = 0.9, Speaker = SpeakerLabel.Unassigned };
        var result = aligner.Align(new AudioBuffer(new float[16000], 16000), new Transcript { Text = "HELLO,", Words = [original] }, "en", TestContext.Current.CancellationToken);
        Assert.Equal(original, result.Words[0] with { Start = null, End = null, TimingMethod = TimingMethod.None });
        Assert.Equal(TimingMethod.ForcedAligner, result.Words[0].TimingMethod);
        var unsplittable = new Transcript { Text = "two words", Words = [new Word { Text = "two words" }] };
        Assert.Null(aligner.Align(new AudioBuffer(new float[16000], 16000), unsplittable, "en", TestContext.Current.CancellationToken).Words[0].Start);
    }

    [Fact]
    public void PriorityIsPerWordAndDoesNotLoadAnUnusedFallback()
    {
        var audio = new AudioBuffer(new float[16000], 16000);
        var original = new Transcript { Text = "one two", Words = [new Word { Text = "one", Confidence = .9 }, new Word { Text = "two", Confidence = .8 }] };
        var first = new PartialAligner();
        var second = new TestAligner(succeeds: true);
        var result = VadTranscriptionPipeline.ApplyTiming(audio, original, "en", [first, second], TestContext.Current.CancellationToken);
        Assert.Equal(TimingMethod.ForcedAligner, result.Words[0].TimingMethod);
        Assert.Equal(TimingMethod.CtcAligner, result.Words[1].TimingMethod);
        Assert.Equal(.9, result.Words[0].Confidence);
        Assert.Equal(.8, result.Words[1].Confidence);
        IEnumerable<Aligner> Lazy()
        {
            yield return second;
            throw new InvalidOperationException("A successful alignment must not load the next model.");
        }
        Assert.All(VadTranscriptionPipeline.ApplyTiming(audio, original, "en", Lazy(), TestContext.Current.CancellationToken).Words, w => Assert.Equal(TimingMethod.CtcAligner, w.TimingMethod));
    }

    [Fact]
    public void ScopedFallbackIteratorReleasesItsLeaseOnEarlySuccessAndFailure()
    {
        var audio = new AudioBuffer(new float[16000], 16000);
        var original = new Transcript { Text = "word", Words = [new Word { Text = "word" }] };
        bool released = false;
        IEnumerable<Aligner> Candidates(Aligner fallback)
        {
            yield return new TestAligner(false);
            try { yield return fallback; }
            finally { released = true; }
        }
        var result = VadTranscriptionPipeline.ApplyTiming(audio, original, "en", Candidates(new TestAligner(true)), TestContext.Current.CancellationToken);
        Assert.Equal(TimingMethod.CtcAligner, result.Words[0].TimingMethod);
        Assert.True(released);
        released = false;
        _ = VadTranscriptionPipeline.ApplyTiming(audio, original, "en", Candidates(new FailingAligner()), TestContext.Current.CancellationToken);
        Assert.True(released);
    }

    [Fact]
    public void ReloadedCtcFallbackHasBitIdenticalWordTimesOnPublicSpeech()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_OMNI_DIRECTORY");
        string? corpus = Environment.GetEnvironmentVariable("ONESPEECH_PROBE_CORPUS");
        Assert.SkipWhen(directory is null || corpus is null, "Set ONESPEECH_OMNI_DIRECTORY and ONESPEECH_PROBE_CORPUS to verified model/public evaluation data.");
        var info = new ModelInfo { Id = "facebook/omniASR-CTC-300M", Source = "https://huggingface.co/facebook/omniASR-CTC-300M", Revision = "8e35f0cc28fa6099e0c14d56db85ce0423baa691", Tier = ModelTier.Tier1, License = "Apache-2.0", Attribution = "Meta, Omnilingual ASR, Apache-2.0." };
        OneSpeech.Alignment.OmniAsrAligner Load() => new(Path.Combine(directory!, "ctc-fp32.onemodel"), Path.Combine(directory!, "omniASR_tokenizer.model"), info, threads: 4);
        using var cached = Load();
        string root = Path.Combine(corpus!, "test-clean", "1089", "134686");
        var text = File.ReadAllLines(Path.Combine(root, "1089-134686.trans.txt"));
        int compared = 0;
        for (int i = 0; i < 5; i++)
        {
            string id = $"1089-134686-{i:D4}";
            var clip = OneAudio.Audio.Load(Path.Combine(root, id + ".flac"), cancellationToken: TestContext.Current.CancellationToken);
            var audio = new AudioBuffer(clip.Samples, 16000);
            string written = text.Single(line => line.StartsWith(id + " ", StringComparison.Ordinal))[(id.Length + 1)..];
            var expected = cached.Align(audio, written, "en", TestContext.Current.CancellationToken);
            using var scoped = Load();
            var actual = scoped.Align(audio, written, "en", TestContext.Current.CancellationToken);
            Assert.Equal(expected.Words, actual.Words);
            compared += actual.Words.Length;
        }
        TestContext.Current.TestOutputHelper!.WriteLine($"CTC cached/scoped timing: {compared} words, zero bitwise differences.");
    }

    [Fact]
    public void WindowFailureFallsBackAndCancellationIsNotSwallowed()
    {
        var audio = new AudioBuffer(new float[16000], 16000);
        var original = new Transcript { Text = "word", Words = [new Word { Text = "word" }] };
        var result = VadTranscriptionPipeline.ApplyTiming(audio, original, "en", [new FailingAligner(), new TestAligner(true)], TestContext.Current.CancellationToken);
        Assert.Equal(TimingMethod.CtcAligner, result.Words[0].TimingMethod);
        Assert.Null(VadTranscriptionPipeline.ApplyTiming(audio, original, "en", [new FailingAligner()], TestContext.Current.CancellationToken).Words[0].Start);
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => VadTranscriptionPipeline.ApplyTiming(audio, original, "en", [new FailingAligner()], canceled.Token));
        Assert.True(OrderedTimingAligner.CanUseQwen(audio, "en"));
        Assert.False(OrderedTimingAligner.CanUseQwen(audio, "ar"));
        Assert.False(OrderedTimingAligner.CanUseQwen(audio, null));
        Assert.False(OrderedTimingAligner.CanUseQwen(new AudioBuffer(new float[200], 16000), "en"));
        Assert.False(OrderedTimingAligner.CanUseQwen(new AudioBuffer(new float[4800001], 16000), "en"));
    }

    [Fact]
    public void ExplicitRealignmentDoesNotReuseOldAudioTimingsWhenNoWindowFits()
    {
        var audio = new AudioBuffer(new float[61 * 16000], 16000);
        var original = new Transcript { Text = "word", Words = [new Word { Text = "word", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native }] };
        using var aligner = new OrderedTimingAligner(new OneSpeech.Catalog.ModelStore(), realign: true);
        var result = aligner.Align(audio, original, "ar", TestContext.Current.CancellationToken);
        Assert.Null(result.Words[0].Start);
        Assert.Null(result.Words[0].End);
        Assert.Equal(TimingMethod.None, result.Words[0].TimingMethod);
        using var pipelineTiming = new OrderedTimingAligner(new OneSpeech.Catalog.ModelStore());
        Assert.Equal(original.Words, pipelineTiming.Align(audio, original, "ar", TestContext.Current.CancellationToken).Words);
    }

    [Fact]
    public void DeferredTimingReplaysTheSameChunksAndProjectsSegmentWords()
    {
        var audio = new AudioBuffer(new float[61 * 16000], 16000);
        var deferred = new DeferredTiming();
        var first = new VadTranscriptionPipeline(new TestRecognizer(false), new TestVad()) { RecognizedChunk = deferred.Add };
        var original = first.Transcribe(audio, "en", TestContext.Current.CancellationToken);
        var timed = deferred.Apply([audio], new TestVad(), original, new TestAligner(true), TestContext.Current.CancellationToken);
        var direct = new VadTranscriptionPipeline(new TestRecognizer(false), new TestVad()) { TimingAligners = [new TestAligner(true)] }.Transcribe(audio, "en", TestContext.Current.CancellationToken);
        Assert.Equal(direct.Text, timed.Text);
        Assert.Equal(direct.Words, timed.Words);
        Assert.Equal(direct.Segments.SelectMany(s => s.Words), timed.Segments.SelectMany(s => s.Words));
        Assert.Throws<InvalidDataException>(() => deferred.Apply([new AudioBuffer(new float[16000], 16000)], new TestVad(), original, new TestAligner(true), TestContext.Current.CancellationToken));
    }

    private sealed class PartialAligner : Aligner
    {
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { ForcedAlignment = true, Languages = ["en"] };
        public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken) => throw new NotSupportedException();
        public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language, CancellationToken cancellationToken) => transcript with
        {
            Words = [transcript.Words[0] with { Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(.1), TimingMethod = TimingMethod.ForcedAligner }, transcript.Words[1]]
        };
    }
    private sealed class FailingAligner : Aligner
    {
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { ForcedAlignment = true, Languages = ["en"] };
        public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken) => throw new UnsupportedAudioException("Constructed window limit.");
    }

    private sealed class TextOnlyAligner : Aligner
    {
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { ForcedAlignment = true, Languages = ["en"] };
        public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken) => new()
        {
            Text = text, Words = text.Split(' ').Select(w => new Word { Text = w, Start = TimeSpan.FromSeconds(0.1), End = TimeSpan.FromSeconds(0.2), TimingMethod = TimingMethod.ForcedAligner }).ToImmutableArray()
        };
    }

    [Fact]
    public void TimingHookPreservesTheRecognizersStreamingPath()
    {
        var pipeline = new VadTranscriptionPipeline(new TestRecognizer(native: false, streamOnly: true), new TestVad()) { TimingAligners = [new TestAligner(succeeds: true)] };
        var segments = pipeline.TranscribeStream(new AudioBuffer(new float[16000], 16000), "en", TestContext.Current.CancellationToken).ToArray();
        Assert.Single(segments);
        Assert.Equal(TimingMethod.CtcAligner, segments[0].Words[0].TimingMethod);
    }

    private sealed class TestVad : VoiceActivityDetector
    {
        public override ModelInfo Model => Info;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken) => [new() { Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(audio.Samples.Length / 16000.0) }];
    }
    private sealed class TestRecognizer(bool native, bool streamOnly = false) : SpeechRecognizer
    {
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { NativeTimestamps = native, Languages = ["en", "fr"] };
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            if (streamOnly) throw new InvalidOperationException("Offline recognition must not replace streaming recognition.");
            var word = new Word { Text = "test", Start = TimeSpan.FromSeconds(0.1), End = TimeSpan.FromSeconds(0.5), TimingMethod = native ? TimingMethod.Native : TimingMethod.Dtw };
            return new Transcript { Text = "test", Words = [word], Segments = [new Segment { Text = "test", Words = [word] }] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => streamOnly ? new TestRecognizer(native).Transcribe(audio, language, cancellationToken).Segments : Transcribe(audio, language, cancellationToken).Segments;
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
    }
    private sealed class TestAligner(bool succeeds) : Aligner
    {
        internal int Calls { get; private set; }
        public override ModelInfo Model => Info;
        public override Capabilities Capabilities => new() { ForcedAlignment = true, Languages = ["en"] };
        public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken) => throw new NotSupportedException();
        public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language, CancellationToken cancellationToken)
        {
            Calls++;
            return transcript with { Words = transcript.Words.Select(w => w with { Start = succeeds ? TimeSpan.FromSeconds(0.2) : null, End = succeeds ? TimeSpan.FromSeconds(0.4) : null, TimingMethod = succeeds ? TimingMethod.CtcAligner : TimingMethod.None }).ToImmutableArray() };
        }
    }
}
