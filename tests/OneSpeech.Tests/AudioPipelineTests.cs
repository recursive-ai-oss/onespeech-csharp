using System.Collections.Immutable;
using OneSpeech.Audio;
using OneSpeech.Pipeline;
using OneSpeech.Vad;
using Xunit;

namespace OneSpeech.Tests;

public sealed class AudioPipelineTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData(7, 255, 0)]
    [InlineData(7, 127, 0)]
    [InlineData(7, 128, 32124)]
    [InlineData(7, 0, -32124)]
    [InlineData(7, 254, 8)]
    [InlineData(7, 126, -8)]
    [InlineData(6, 213, 8)]
    [InlineData(6, 85, -8)]
    [InlineData(6, 170, 32256)]
    [InlineData(6, 42, -32256)]
    public void G711PublishedValues(int tag, byte code, int expected)
    {
        foreach (bool extensible in new[] { false, true })
        {
            using var stream = Wave(tag, [code, code], extensible);
            using var source = AudioFileSource.Open(stream, true, Token);
            var pcm = new float[8];
            Assert.Equal(2, source.Read(pcm, Token));
            Assert.Equal(expected / 32768f, pcm[0]);
            Assert.Equal(0, source.Read(pcm, Token));
        }
    }

    [Fact]
    public void ExtensibleCompressedSamplesPerBlockIsAcceptedAndBounded()
    {
        foreach (byte samplesPerBlock in new byte[] { 0, 1, 8 })
        {
            using var original = Wave(6, [213, 213], true);
            byte[] bytes = original.ToArray();
            bytes[38] = samplesPerBlock;
            using var stream = new MemoryStream(bytes);
            using var source = AudioFileSource.Open(stream, true, Token);
            Assert.Equal(2, source.Read(new float[2], Token));
        }
        using var valid = Wave(7, [255, 255], true);
        byte[] invalid = valid.ToArray();
        invalid[36] = 255;
        using var truncatedExtension = new MemoryStream(invalid);
        Assert.Throws<UnsupportedAudioException>(() => AudioFileSource.Open(truncatedExtension, true, Token));
    }

    [Fact]
    public void LookbehindRetainsShortContinuousSpeechAfterForcedCut()
    {
        var chunks = new VadChunker(new AllSpeech { Minimum = TimeSpan.FromMilliseconds(250) }, TimeSpan.FromSeconds(1))
            .Read([new AudioBuffer(new float[33600], 16000)], null, Token).ToArray();
        Assert.Equal(3, chunks.Length);
        Assert.Equal(33600, chunks.Sum(c => c.Audio.FrameCount));
        Assert.Equal(TimeSpan.FromSeconds(2), chunks[^1].Start);
        Assert.Equal(1600, chunks[^1].Audio.FrameCount);
    }

    [Fact]
    public void G711RejectsTruncationAndInvalidFormats()
    {
        using var original = Wave(7, [255, 255]);
        byte[] valid = original.ToArray();
        for (int length = 0; length < valid.Length; length++)
        {
            using var truncated = new MemoryStream(valid[..length]);
            Assert.Throws<UnsupportedAudioException>(() => AudioFileSource.Open(truncated, true, Token));
        }
        foreach (int field in new[] { 22, 32, 34 })
        {
            byte[] invalid = (byte[])valid.Clone();
            invalid[field] = 0;
            using var stream = new MemoryStream(invalid);
            Assert.Throws<UnsupportedAudioException>(() => AudioFileSource.Open(stream, true, Token));
        }
    }

    [Fact]
    public void ResamplingPreservesChannelsAndDuration()
    {
        using var stream = Wave(7, Enumerable.Repeat((byte)255, 8000).ToArray(), rate: 8000);
        using var source = AudioFileSource.Open(stream, true, Token);
        var pcm = new float[333];
        int frames, total = 0;
        while ((frames = source.Read(pcm, Token)) != 0) { total += frames; Assert.All(pcm[..frames], x => Assert.Equal(0, x)); }
        Assert.Equal(16000, total);
    }

    [Fact]
    public void PipelineShiftsAllTimingsAndReportsAudioTime()
    {
        var progress = new ProgressSink();
        var pipeline = new VadTranscriptionPipeline(new FakeRecognizer(), new AllSpeech(), TimeSpan.FromSeconds(1), progress);
        var result = pipeline.Transcribe(new AudioBuffer(new float[5 * 16000 * 2], 16000, 2), null, Token);
        Assert.Equal(TimeSpan.FromSeconds(5), result.Duration);
        Assert.Equal(5, result.Segments.Length);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(TimeSpan.FromSeconds(i), result.Words[i * 2].Start);
            Assert.Equal(TimeSpan.FromSeconds(i + 0.5), result.Segments[i].Words[0].End);
            Assert.Null(result.Words[i * 2 + 1].Start);
        }
        Assert.Equal("en", result.Language);
        Assert.Equal(2, result.Models.Length);
        Assert.Equal(progress.Values.Order(), progress.Values);
        Assert.Equal(TimeSpan.FromSeconds(5), progress.Values[^1]);
    }

    [Fact]
    public void CancellationStopsBeforeNextChunk()
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var recognizer = new FakeRecognizer { OnCall = cancellation.Cancel };
        var pipeline = new VadTranscriptionPipeline(recognizer, new AllSpeech(), TimeSpan.FromSeconds(1));
        Assert.Throws<OperationCanceledException>(() => pipeline.Transcribe(new AudioBuffer(new float[5 * 16000], 16000), null, cancellation.Token));
        Assert.Equal(1, recognizer.Calls);
    }

    [Fact]
    public async Task AsyncInputIsNotOwnedAndUsesFileClock()
    {
        using var input = Wave(7, Enumerable.Repeat((byte)255, 48000).ToArray());
        var pipeline = new VadTranscriptionPipeline(new FakeRecognizer(), new AllSpeech(), TimeSpan.FromSeconds(1));
        var result = await pipeline.TranscribeAsync(input, null, Token);
        Assert.True(input.CanRead);
        Assert.Equal(TimeSpan.FromSeconds(3), result.Duration);
        Assert.Equal(TimeSpan.FromSeconds(2), result.Segments[^1].Start);
    }

    [Fact]
    public void EmbeddedVadDetectsSilenceAtBothRates()
    {
        using var vad = new SileroVad(cancellationToken: Token);
        foreach (int rate in new[] { 8000, 16000 })
        {
            Assert.Empty(vad.Detect(new AudioBuffer(new float[rate], rate), Token));
            Assert.Empty(vad.Detect(new AudioBuffer(ReadOnlyMemory<float>.Empty, rate), Token));
        }
    }

    // All fixtures are generated here: no recorded voice, external codec output, or borrowed bytes.
    internal static MemoryStream Wave(int tag, byte[] payload, bool extensible = false, int rate = 16000)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            int formatSize = extensible ? 40 : 16;
            writer.Write("RIFF"u8); writer.Write(20 + formatSize + payload.Length + (payload.Length & 1)); writer.Write("WAVEfmt "u8);
            writer.Write(formatSize); writer.Write((ushort)(extensible ? 65534 : tag)); writer.Write((ushort)1);
            writer.Write(rate); writer.Write(rate); writer.Write((ushort)1); writer.Write((ushort)8);
            if (extensible)
            {
                writer.Write((ushort)22); writer.Write((ushort)8); writer.Write(0);
                writer.Write(new Guid(tag, 0, 16, 128, 0, 0, 170, 0, 56, 155, 113).ToByteArray());
            }
            writer.Write("data"u8); writer.Write(payload.Length); writer.Write(payload); if ((payload.Length & 1) != 0) writer.Write((byte)0);
        }
        stream.Position = 0;
        return stream;
    }

    private static ModelInfo Model(string id) => new() { Id = id, Source = "https://example.org/model", Revision = "test", License = "MIT", Attribution = "Synthetic fixture", Tier = ModelTier.Tier1 };
    private sealed class AllSpeech : VoiceActivityDetector
    {
        public TimeSpan Minimum { get; init; }
        public override ModelInfo Model { get; } = AudioPipelineTests.Model("vad");
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken) => audio.Duration < Minimum ? [] : [new() { Start = TimeSpan.Zero, End = audio.Duration }];
    }
    private sealed class FakeRecognizer : SpeechRecognizer
    {
        public Action? OnCall { get; init; }
        public int Calls { get; private set; }
        public override ModelInfo Model { get; } = AudioPipelineTests.Model("recognizer");
        public override Capabilities Capabilities { get; } = new() { Languages = ["en"] };
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            Assert.Equal(1, audio.Channels);
            Calls++; OnCall?.Invoke();
            var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(0.5), TimingMethod = TimingMethod.Native };
            return new Transcript { Text = "test", Words = [word, new Word { Text = "untimed" }], Segments = [new Segment { Text = "test", Words = [word], Start = TimeSpan.Zero, End = audio.Duration }], Language = "en", Models = [Model] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => Transcribe(audio, language, cancellationToken).Segments;
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
    }
    private sealed class ProgressSink : IProgress<TimeSpan>
    {
        public List<TimeSpan> Values { get; } = [];
        public void Report(TimeSpan value) => Values.Add(value);
    }
}
