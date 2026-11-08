using System.Collections.Immutable;
using System.Text;
using OneSpeech.Alignment;
using OneSpeech.Text;
using Xunit;

namespace OneSpeech.Tests;

public sealed class OmniAlignmentTests
{
    [Theory]
    [InlineData("Hello,", "hello")]
    [InlineData("42", "forty two")]
    [InlineData("1,234", "one thousand two hundred thirty four")]
    [InlineData("0", "zero")]
    [InlineData("One-Word!", "one word")]
    [InlineData("世界。", "世界")]
    [InlineData("12.50", "\uFFFD")]
    [InlineData("$12", "\uFFFD")]
    [InlineData("a--b", "a b")]
    [InlineData("Cafe\u0301,", "café")]
    [InlineData("It’s", "it's")]
    public void SpokenExpansionDoesNotChangeTheWrittenForm(string written, string spoken) =>
        Assert.Equal(spoken, OmniAsrAligner.SpokenForm(written, "en"));

    [Fact]
    public void CharacterReaderRejectsMalformedAndOverlongFields()
    {
        Assert.Throws<InvalidDataException>(() => SentencePieceProto.Read([10, 255, 255, 255, 255, 15]));
        Assert.Throws<InvalidDataException>(() => SentencePieceProto.Read([10, 3, 10, 1, 255]));
        Assert.Throws<InvalidDataException>(() => SentencePieceProto.Read([0]));
        Assert.Throws<InvalidDataException>(() => SentencePieceProto.Read(new byte[SentencePieceProto.MaxFileBytes + 1]));
    }

    [Fact]
    public void ExplicitWordsKeepCasingPunctuationConfidenceAndSpeaker()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ctc-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string vocabulary = Path.Combine(directory, "vocabulary.model");
            WriteVocabulary(vocabulary);
            var g = new SpeakerGraph();
            float[] logits = new float[6 * 9812]; Array.Fill(logits, -15);
            for (int frame = 0; frame < 6; frame++) logits[frame * 9812 + (frame is 2 or 3 ? 5 : 0)] = 0;
            string output = g.Float("emissions", [1, 6, 9812], logits);
            string graph = Path.Combine(directory, "ctc.onemodel");
            g.Save(graph, output, 9812, "waveform", [1, 1, "samples"], "logits", [1, 6, 9812]);
            var model = new ModelInfo { Id = "synthetic", Source = "synthetic", Revision = "1", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Synthetic lattice, no speech weights." };
            using var aligner = new OmniAsrAligner(graph, vocabulary, model, cancellationToken: TestContext.Current.CancellationToken);
            var originalWord = new Word { Text = "A!", Confidence = 0.75, Language = "en", Speaker = new SpeakerLabel { Id = "one" }, Overlap = true };
            var original = new Transcript { Text = "A!", Words = [originalWord], Segments = [new Segment { Text = "A!", Words = [originalWord] }] };
            var result = aligner.Align(new AudioBuffer(new float[2000], 16000), original, "en", TestContext.Current.CancellationToken);
            Assert.Equal("A!", result.Text);
            Assert.Equal(TimeSpan.FromSeconds(0.04), result.Words[0].Start);
            Assert.Equal(TimeSpan.FromSeconds(0.08), result.Words[0].End);
            Assert.Equal(TimingMethod.CtcAligner, result.Words[0].TimingMethod);
            Assert.Equal(originalWord, result.Words[0] with { Start = null, End = null, TimingMethod = TimingMethod.None });
            Assert.Equal(result.Words, result.Segments[0].Words);
            Assert.Null(originalWord.Start);
            var shortAudio = aligner.Align(new AudioBuffer(new float[10], 16000), original, "en", TestContext.Current.CancellationToken);
            Assert.Null(shortAudio.Words[0].Start);
            Assert.Null(shortAudio.Words[0].End);
            Assert.Throws<InvalidDataException>(() => aligner.Align(new AudioBuffer(new float[2400], 16000), original, "en", TestContext.Current.CancellationToken));
            var oversizedTarget = original with { Words = Enumerable.Repeat(originalWord, 2000).ToImmutableArray() };
            Assert.Throws<CapabilityNotSupportedException>(() => aligner.Align(new AudioBuffer(new float[960000], 16000), oversizedTarget, "en", TestContext.Current.CancellationToken));
            Assert.Throws<CapabilityNotSupportedException>(() => aligner.Align(new AudioBuffer(ReadOnlyMemory<float>.Empty, 16000), original, "unsupported", TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task EncodedStreamsUseAsyncReadsAndRemainCallerOwnedOnFailure()
    {
        using var stream = new AsyncOnlyStream();
        await Assert.ThrowsAsync<UnsupportedAudioException>(() => Speech.AlignAsync(stream, "test", TestContext.Current.CancellationToken).AsTask());
        Assert.True(stream.CanRead);
        Assert.True(stream.ReadCalls > 0);
    }

    [Fact]
    public async Task EncodedAlignmentCancellationPrecedesModelAcquisition()
    {
        using var source = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        source.Cancel();
        using var stream = new AsyncOnlyStream();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Speech.AlignAsync(stream, "test", source.Token).AsTask());
        Assert.Equal(0, stream.ReadCalls);
    }

    private sealed class AsyncOnlyStream : Stream
    {
        internal int ReadCalls { get; private set; }
        private bool disposed;
        public override bool CanRead => !disposed;
        protected override void Dispose(bool disposing) { disposed = true; base.Dispose(disposing); }
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ReadCalls++;
            if (ReadCalls > 1) return ValueTask.FromResult(0);
            buffer.Span[0] = 1; return ValueTask.FromResult(1);
        }
        public override int Read(byte[] buffer, int offset, int count) => throw new InvalidOperationException("Synchronous caller-stream read.");
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }

    internal static void WriteVocabulary(string path)
    {
        using var file = File.Create(path);
        for (int id = 0; id < 9812; id++)
        {
            string piece = id switch { 0 => "<s>", 1 => "<pad>", 2 => "</s>", 3 => "<unk>", 4 => " ", 5 => "a", _ => new Rune(0x1000 + id).ToString() };
            using var field = new MemoryStream();
            field.WriteByte(10); byte[] text = Encoding.UTF8.GetBytes(piece); Varint(field, text.Length); field.Write(text);
            field.WriteByte(24); Varint(field, id < 3 ? 3 : id == 3 ? 2 : 1);
            file.WriteByte(10); Varint(file, (int)field.Length); field.Position = 0; field.CopyTo(file);
        }
        file.Write([18, 3, 0xc0, 0x02, 3]); // TrainerSpec: unknown ID 3.
        file.Write([26, 4, 24, 0, 40, 0]); // NormalizerSpec: no dummy prefix, no escaped whitespace.
    }
    private static void Varint(Stream stream, int value)
    {
        while (value >= 128) { stream.WriteByte((byte)((value & 127) | 128)); value >>= 7; }
        stream.WriteByte((byte)value);
    }
}
