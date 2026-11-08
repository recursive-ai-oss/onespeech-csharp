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
