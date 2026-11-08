using OneSpeech.Audio;
using OneSpeech.Media;
using RecursiveAI.OneMedia;
using RecursiveAI.OneMedia.Captions;
using Xunit;

namespace OneSpeech.Media.Tests;

public sealed class ContainerTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Fact]
    public void MatroskaAudioIsDecodedAndCallerStreamRemainsOpen()
    {
        MediaInput.Register();
        using var aac = ContainerFixture.Aac();
        using var mkv = ContainerFixture.Matroska(aac, Token);
        using (var media = RecursiveAI.OneMedia.Media.Open(mkv, leaveOpen: true, Token))
        {
            Assert.Equal(TrackKind.Video, media.Tracks[0].Kind);
            Assert.NotNull(media.Audio);
        }
        mkv.Position = 0;
        using (var source = AudioFileSource.Open(mkv, true, Token))
        {
            Assert.Equal(1, source.Channels);
            float[] buffer = new float[4096];
            int frames, total = 0;
            while ((frames = source.Read(buffer, Token)) > 0) total += frames;
            Assert.InRange(total, 15000, 18000);
        }
        Assert.True(mkv.CanRead);
        mkv.Position = 0;
        using (var owned = AudioFileSource.Open(mkv, false, Token)) Assert.True(owned.Read(new float[100], Token) > 0);
        Assert.False(mkv.CanRead);
    }

    [Fact]
    public void CaptionsMuxIntoMp4AndReadBackWithAudioIntact()
    {
        string root = Path.Combine(Path.GetTempPath(), "media-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        string input = Path.Combine(root, "input.mp4"), output = Path.Combine(root, "output.mp4");
        try
        {
            using (var aac = ContainerFixture.Aac()) File.WriteAllBytes(input, aac.ToArray());
            var report = CaptionTests.Example().MuxCaptions(input, output, Token);
            Assert.NotEmpty(report.Packets);
            using var media = RecursiveAI.OneMedia.Media.Open(output, Token);
            Assert.NotNull(media.Audio);
            var captions = Assert.Single(media.ContainerSubtitles).Document;
            Assert.Equal("Hello world", Assert.Single(captions.Cues).Text);
            Assert.Contains(captions.Cues[0].Spans, span => span.Voice == "alice");
            Assert.Throws<IOException>(() => CaptionTests.Example().MuxCaptions(input, output, Token));
            Assert.True(File.Exists(output));
        }
        finally { Directory.Delete(root, true); }
    }

    [Fact]
    public void TrackEncodingCanFeedMatroskaMuxer()
    {
        using var output = new MemoryStream();
        var track = CaptionTests.Example().EncodeCaptionTrack(TextTrackFormat.WebVtt, Token);
        using (var muxer = new RecursiveAI.OneMedia.Matroska.MatroskaMuxer(output))
        {
            int index = muxer.AddTrack(track.Track);
            foreach (var packet in track.Packets) muxer.WritePacket(packet.Info with { TrackIndex = index }, packet.Data, Token);
            muxer.Finish(Token);
        }
        output.Position = 0;
        using var media = RecursiveAI.OneMedia.Media.Open(output, leaveOpen: true, Token);
        Assert.Equal("Hello world", Assert.Single(Assert.Single(media.ContainerSubtitles).Document.Cues).Text);
    }

    private sealed class NonseekableStream(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => inner.ReadAsync(buffer, cancellationToken);
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Fact]
    public async Task InvalidStreamFailsBeforeModelsAndDoesNotCloseInput()
    {
        MediaInput.Register();
        using var bytes = new MemoryStream("not a container"u8.ToArray());
        using var stream = new NonseekableStream(bytes);
        await Assert.ThrowsAnyAsync<MediaException>(async () =>
        {
            await foreach (var _ in MediaInput.TranscribeStreamAsync(stream, cancellationToken: Token)) { }
        });
        Assert.True(bytes.CanRead);
    }

    [Fact]
    public async Task SpeechLevelTranscribesMatroskaAudioTrack()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("ONESPEECH_MEDIA_MODEL_SMOKE") == "1", "Enable the model integration smoke with a populated model cache.");
        MediaInput.Register();
        string path = Path.Combine(Path.GetTempPath(), "speech-media-" + Guid.NewGuid().ToString("N") + ".mkv");
        try
        {
            using var aac = ContainerFixture.Aac(Path.Combine(AppContext.BaseDirectory, "Fixtures/1089-134686-0001.flac"));
            using var mkv = ContainerFixture.Matroska(aac, Token);
            File.WriteAllBytes(path, mkv.ToArray());
            var transcript = Speech.Transcribe(path, Token, language: "en", model: "openai/whisper-tiny");
            Assert.Contains("stuff", transcript.Text, StringComparison.OrdinalIgnoreCase);
            Assert.NotEmpty(transcript.Words);
            Assert.NotEmpty(transcript.Turns);
            var segments = new List<Segment>();
            await foreach (var segment in Speech.TranscribeStream(path, Token, language: "en", model: "openai/whisper-tiny")) segments.Add(segment);
            Assert.NotEmpty(segments);
            // The asynchronous stream path spools nonseekable input without taking ownership.
            mkv.Position = 0;
            using var nonseekable = new NonseekableStream(mkv);
            var fromStream = await Speech.TranscribeAsync(nonseekable, Token, language: "en", model: "openai/whisper-tiny");
            Assert.Equal(transcript.Text, fromStream.Text);
            Assert.True(mkv.CanRead);
            mkv.Position = 0;
            var streamedSegments = new List<Segment>();
            await foreach (var segment in MediaInput.TranscribeStreamAsync(nonseekable, "en", "openai/whisper-tiny", Token)) streamedSegments.Add(segment);
            Assert.Equal(segments.Select(s => s.Text), streamedSegments.Select(s => s.Text));
            var aligned = Speech.Align(path, transcript.Text, Token, language: "en");
            Assert.NotEmpty(aligned.Words);
            Assert.Contains(aligned.Words, w => w.Start is not null);
        }
        finally { File.Delete(path); }
    }
}
