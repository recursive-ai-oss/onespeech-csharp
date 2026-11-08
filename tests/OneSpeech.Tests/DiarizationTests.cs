using System.Collections.Immutable;
using OneSpeech.Diarization;
using OneSpeech.Pipeline;
using Xunit;

namespace OneSpeech.Tests;

public sealed class DiarizationTests
{
    private static readonly string[] ExclusiveIds = ["b", "a", "b"];
    private static readonly string[] ChannelIds = ["channel-1", "channel-2"];
    private static ActiveTurn T(string id, double start, double end, double activity = 1) => new()
    { Turn = new Turn { Speaker = new SpeakerLabel { Id = id }, Start = TimeSpan.FromSeconds(start), End = TimeSpan.FromSeconds(end), Text = "" }, Activity = activity };
    private static Word W(double? start, double? end) => new() { Text = " word", Start = start is {} s ? TimeSpan.FromSeconds(s) : null, End = end is {} e ? TimeSpan.FromSeconds(e) : null };

    [Fact]
    public void MajorityOverlapAndMissingTimes()
    {
        var words = ImmutableArray.Create(W(0, 3), W(5, 6), W(null, null), W(2, 2));
        var transcript = new Transcript { Text = "unchanged", Words = words, Segments = [new Segment { Text = "unchanged", Words = words }] };
        var result = TranscriptDiarization.Merge(transcript, [T("a", 0, 1), T("b", 1, 3)], TestContext.Current.CancellationToken);
        Assert.Equal("b", result.Words[0].Speaker!.Id);
        Assert.Equal(result.Words[0], Assert.Single(result.Turns.Single(t => t.Speaker.Id == "b").Words));
        Assert.False(result.Words[0].Overlap);
        Assert.All(result.Words.Skip(1), w => Assert.Equal(SpeakerLabel.Unassigned, w.Speaker));
        Assert.Equal(result.Words, result.Segments[0].Words);
        Assert.Equal("unchanged", result.Text);
        Assert.Null(transcript.Words[0].Speaker);
        var overlap = TranscriptDiarization.Merge(new Transcript { Text = "", Words = [W(1, 2)] }, [T("a", 0, 3, 0.2), T("b", 1, 2, 0.8)], TestContext.Current.CancellationToken);
        Assert.Equal("b", overlap.Words[0].Speaker!.Id);
        Assert.True(overlap.Words[0].Overlap);
    }

    [Fact]
    public void ExclusiveSplitsAndCoalescesWithStableTies()
    {
        var turns = TranscriptDiarization.Exclusive([T("b", 0, 4, 0.5), T("a", 1, 2, 0.5), T("b", 2, 3, 0.5)], TestContext.Current.CancellationToken);
        Assert.Equal(ExclusiveIds, turns.Select(t => t.Speaker.Id));
        Assert.Equal(TimeSpan.FromSeconds(4), turns[^1].End);
        Assert.Throws<ArgumentException>(() => TranscriptDiarization.Exclusive([T("x", 2, 1)], TestContext.Current.CancellationToken));
        using var cts = new CancellationTokenSource(); cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => TranscriptDiarization.Exclusive([T("x", 0, 1)], cts.Token));
    }

    private static AudioBuffer Stereo(double leakage)
    {
        var pcm = new float[32000 * 2];
        for (int i = 0; i < 32000; i++)
        {
            float value = (float)(0.5 * Math.Sin(2 * Math.PI * 440 * i / 16000));
            pcm[2 * i] = i < 16000 ? value : (float)(leakage * value);
            pcm[2 * i + 1] = i >= 16000 ? value : (float)(leakage * value);
        }
        return new AudioBuffer(pcm, 16000, 2);
    }

    [Fact]
    public void ChannelDetectionAcceptsSeparatedAndRejectsLeakyMix()
    {
        Assert.True(ChannelDiarizer.IsSeparated(Stereo(0), TestContext.Current.CancellationToken));
        Assert.True(ChannelDiarizer.IsSeparated(Stereo(0.01), TestContext.Current.CancellationToken));
        Assert.False(ChannelDiarizer.IsSeparated(Stereo(0.3), TestContext.Current.CancellationToken));
        Assert.False(ChannelDiarizer.IsSeparated(new AudioBuffer(new float[3200], 16000, 2), TestContext.Current.CancellationToken));
        var detector = new EnergyVad();
        var turns = new ChannelDiarizer(detector).Diarize(Stereo(0), TestContext.Current.CancellationToken);
        Assert.Equal(2, turns.Length);
        Assert.Equal("channel-1", turns[0].Speaker.Id);
        Assert.Equal("channel-2", turns[1].Speaker.Id);
        Assert.Throws<CapabilityNotSupportedException>(() => new ChannelDiarizer(detector).Diarize(Stereo(0.3), TestContext.Current.CancellationToken));
        Assert.NotEmpty(new ChannelDiarizer(detector, true).Diarize(Stereo(0.3), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void PipelinePreservesOriginalChannelsAndAssignsWords()
    {
        var detector = new EnergyVad();
        var pipeline = new VadTranscriptionPipeline(new FakeRecognizer(), detector, null, null, new ChannelDiarizer(detector));
        var transcript = pipeline.Transcribe(Stereo(0), "en", TestContext.Current.CancellationToken);
        Assert.Equal(2, transcript.Turns.Length);
        Assert.Equal(ChannelIds, transcript.Words.Select(w => w.Speaker!.Id));
        var segments = pipeline.TranscribeStream(Stereo(0), "en", TestContext.Current.CancellationToken).ToArray();
        Assert.Equal(transcript.Words.Select(w => w.Speaker), segments.SelectMany(s => s.Words).Select(w => w.Speaker));

    }

    [Fact]
    public async Task StereoFileAndEncodedStreamHaveGlobalChannelSpeakers()
    {
        var detector = new EnergyVad();
        var pipeline = new VadTranscriptionPipeline(new FakeRecognizer(), detector, null, null, new ChannelDiarizer(detector));
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".wav");
        try
        {
            var audio = Stereo(0);
            using (var file = File.Create(path))
            using (var writer = new BinaryWriter(file))
            {
                writer.Write("RIFF"u8); writer.Write(36 + audio.Samples.Length * 2); writer.Write("WAVEfmt "u8); writer.Write(16);
                writer.Write((short)1); writer.Write((short)2); writer.Write(16000); writer.Write(64000); writer.Write((short)4); writer.Write((short)16);
                writer.Write("data"u8); writer.Write(audio.Samples.Length * 2);
                foreach (float sample in audio.Samples.ToArray()) writer.Write((short)(sample * 32767));
            }
            var result = pipeline.Transcribe(path, "en", TestContext.Current.CancellationToken);
            Assert.Equal(ChannelIds, result.Words.Select(w => w.Speaker!.Id));
            Assert.Equal(2, result.Turns.Length);
            Assert.Equal(TimeSpan.FromSeconds(2), result.Duration);
            Assert.Equal(result.Words.Select(w => w.Speaker), pipeline.TranscribeStream(path, "en", TestContext.Current.CancellationToken).SelectMany(s => s.Words).Select(w => w.Speaker));
            await using var encoded = File.OpenRead(path);
            var fromStream = await pipeline.TranscribeAsync(encoded, "en", TestContext.Current.CancellationToken);
            Assert.Equal(result.Words, fromStream.Words);
        }
        finally { File.Delete(path); }
    }

    private static readonly ModelInfo TestModel = new() { Id = "synthetic", Source = "synthetic", Revision = "1", Tier = ModelTier.Tier1, License = "MIT", Attribution = "synthetic" };
    private sealed class EnergyVad : VoiceActivityDetector
    {
        public override ModelInfo Model => TestModel;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
        {
            var result = ImmutableArray.CreateBuilder<SpeechInterval>(); int first = -1;
            for (int i = 0; i < audio.FrameCount; i += 160)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double energy = 0;
                for (int j = i; j < Math.Min(i + 160, audio.FrameCount); j++) energy += audio.Samples.Span[j] * audio.Samples.Span[j];
                if (energy > 0.01 && first < 0) first = i;
                if (energy <= 0.01 && first >= 0) { result.Add(new SpeechInterval { Start = TimeSpan.FromSeconds((double)first / audio.SampleRate), End = TimeSpan.FromSeconds((double)i / audio.SampleRate) }); first = -1; }
            }
            if (first >= 0) result.Add(new SpeechInterval { Start = TimeSpan.FromSeconds((double)first / audio.SampleRate), End = audio.Duration });
            return result.ToImmutable();
        }
    }
    private sealed class FakeRecognizer : SpeechRecognizer
    {
        public override ModelInfo Model => TestModel;
        public override Capabilities Capabilities { get; } = new() { Languages = ["en"] };
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            var words = ImmutableArray.Create(W(0.1, 0.5), W(1.1, 1.5));
            return new Transcript { Text = " word word", Words = words, Segments = [new Segment { Text = " word word", Words = words }] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => Transcribe(audio, language, cancellationToken).Segments;
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
    }
}
