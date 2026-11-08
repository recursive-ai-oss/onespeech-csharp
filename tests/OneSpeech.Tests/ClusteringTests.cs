using System.Collections.Immutable;
using OneSpeech.Diarization;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ClusteringTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;
    private static readonly ModelInfo Model = new() { Id = "test", Revision = "1", Source = "synthetic", License = "MIT", Attribution = "synthetic", Tier = ModelTier.Tier1 };

    private static AudioBuffer Probe(int repetitions = 3)
    {
        // Flat levels are numerical probes, not synthesized voices or biometric speaker vectors.
        var pcm = new float[repetitions * 3 * 16000];
        for (int region = 0; region < repetitions; region++) Array.Fill(pcm, 0.2f * (region % 3 + 1), region * 3 * 16000, 2 * 16000);
        return new AudioBuffer(pcm, 16000);
    }

    [Theory]
    [InlineData(ClusteringMethod.Spectral)]
    [InlineData(ClusteringMethod.Agglomerative)]
    public void ThreeKnownGroupsHaveStableLabelsAndNoOverlap(ClusteringMethod method)
    {
        using var embedder = new ProbeEmbedder();
        var diarizer = new ClusteringDiarizer(embedder, new ProbeVad(), new ClusteringOptions { Method = method, WindowSeconds = 1, ShiftSeconds = 1 });
        var turns = diarizer.Diarize(Probe(), Token);
        Assert.Equal(3, turns.Length);
        Assert.Equal(3, turns.Select(t => t.Speaker.Id).Distinct().Count());
        Assert.Equal(TimeSpan.Zero, turns[0].Start);
        Assert.Equal(TimeSpan.FromSeconds(8), turns[^1].End);
        Assert.Equal(TimeSpan.FromSeconds(2), turns[0].End);
        Assert.All(turns, t => Assert.Equal(TimeSpan.FromSeconds(2), t.End - t.Start));
        Assert.Equal(6, embedder.Calls);
    }

    [Theory]
    [InlineData(ClusteringMethod.Spectral, 1)]
    [InlineData(ClusteringMethod.Spectral, 2)]
    [InlineData(ClusteringMethod.Agglomerative, 2)]
    public void ExactSpeakerHintsAreHonoured(ClusteringMethod method, int count)
    {
        using var embedder = new ProbeEmbedder();
        var diarizer = new ClusteringDiarizer(embedder, new ProbeVad(), new ClusteringOptions { Method = method, NumSpeakers = count, WindowSeconds = 1, ShiftSeconds = 1 });
        Assert.Equal(count, diarizer.Diarize(Probe(), Token).Select(t => t.Speaker.Id).Distinct().Count());
    }

    [Fact]
    public void MinMaxHintsLandmarksAndReusedBlockStorage()
    {
        using var embedder = new ProbeEmbedder();
        var diarizer = new ClusteringDiarizer(embedder, new ProbeVad(), new ClusteringOptions { MinSpeakers = 2, MaxSpeakers = 2, MaxLandmarks = 16, WindowSeconds = 1, ShiftSeconds = 1 });
        var probe = Probe(24);
        var first = diarizer.Diarize(probe, Token);
        IEnumerable<AudioBuffer> Blocks()
        {
            var buffer = new float[777];
            for (int offset = 0; offset < probe.FrameCount; offset += buffer.Length)
            {
                int length = Math.Min(buffer.Length, probe.FrameCount - offset);
                probe.Samples.Slice(offset, length).CopyTo(buffer);
                yield return new AudioBuffer(buffer.AsMemory(0, length), 16000);
            }
        }
        Assert.Equal(first, diarizer.DiarizeBlocks(Blocks(), Token));
        Assert.Equal(2, first.Select(t => t.Speaker.Id).Distinct().Count());
        Assert.Equal(TimeSpan.FromSeconds(71), first[^1].End);
    }

    [Fact]
    public void SilenceCancellationLimitsAndMalformedEmbeddingsAreRefused()
    {
        using var embedder = new ProbeEmbedder();
        var detector = new ProbeVad();
        Assert.Empty(new ClusteringDiarizer(embedder, detector).Diarize(new AudioBuffer(new float[16000], 16000), Token));
        Assert.Throws<ArgumentException>(() => new ClusteringDiarizer(embedder, detector, new ClusteringOptions { MinSpeakers = 3, MaxSpeakers = 2 }));
        Assert.Throws<CapabilityNotSupportedException>(() => new ClusteringDiarizer(embedder, detector, new ClusteringOptions { MaxWindows = 1 }).Diarize(Probe(), Token));
        Assert.Throws<CapabilityNotSupportedException>(() => new ClusteringDiarizer(embedder, detector, new ClusteringOptions { NumSpeakers = 8 }).Diarize(new AudioBuffer(Enumerable.Repeat(0.2f, 1600).ToArray(), 16000), Token));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => new ClusteringDiarizer(embedder, detector).Diarize(Probe(), cancelled.Token));
        using var invalid = new ProbeEmbedder { Invalid = true };
        Assert.Throws<InvalidOperationException>(() => new ClusteringDiarizer(invalid, detector).Diarize(Probe(), Token));
    }

    [Fact]
    public void IdenticalEmbeddingsDoNotInventExtraSpeakers()
    {
        using var embedder = new ProbeEmbedder();
        var audio = new AudioBuffer(Enumerable.Repeat(0.2f, 10 * 16000).ToArray(), 16000);
        Assert.Single(new ClusteringDiarizer(embedder, new ProbeVad()).Diarize(audio, Token).Select(t => t.Speaker.Id).Distinct());
    }

    [Fact]
    public void ResegmentationSeamValidatesBackendOutput()
    {
        using var embedder = new ProbeEmbedder(); var backend = new PassThrough();
        var diarizer = new ClusteringDiarizer(embedder, new ProbeVad(), new ClusteringOptions { NumSpeakers = 2 }, backend);
        Assert.NotEmpty(diarizer.Diarize(Probe(), Token)); Assert.Equal(1, backend.Calls);
        backend.Invalid = true;
        Assert.Throws<InvalidOperationException>(() => diarizer.Diarize(Probe(), Token));
    }

    private sealed class PassThrough : EmbeddingResegmenter
    {
        internal int Calls; internal bool Invalid;
        public override int[] Resegment(IReadOnlyList<EmbeddingWindow> windows, ReadOnlySpan<int> initialLabels, CancellationToken cancellationToken)
        { cancellationToken.ThrowIfCancellationRequested(); Calls++; Assert.Equal(windows.Count, initialLabels.Length); return Invalid ? [] : initialLabels.ToArray(); }
    }

    private sealed class ProbeEmbedder : SpeakerEmbedder
    {
        internal int Calls; internal bool Invalid;
        public override ModelInfo Model => ClusteringTests.Model;
        public override int Dimension => 3;
        public override int SampleRate => 16000;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); Calls++;
            double mean = 0; foreach (float value in audio.Samples.Span) mean += value;
            int index = Math.Clamp((int)Math.Round(mean / audio.FrameCount / 0.2) - 1, 0, 2);
            destination.Clear(); destination[index] = Invalid ? float.NaN : 1;
        }
        public override void Dispose() { }
    }

    private sealed class ProbeVad : VoiceActivityDetector
    {
        public override ModelInfo Model => ClusteringTests.Model;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
        {
            var intervals = ImmutableArray.CreateBuilder<SpeechInterval>(); int start = -1;
            for (int frame = 0; frame < audio.FrameCount; frame++)
            {
                if (frame % 16000 == 0) cancellationToken.ThrowIfCancellationRequested();
                bool active = audio.Samples.Span[frame] > 0.1;
                if (active && start < 0) start = frame;
                if (!active && start >= 0) { intervals.Add(new SpeechInterval { Start = TimeSpan.FromSeconds(start / 16000.0), End = TimeSpan.FromSeconds(frame / 16000.0) }); start = -1; }
            }
            if (start >= 0) intervals.Add(new SpeechInterval { Start = TimeSpan.FromSeconds(start / 16000.0), End = audio.Duration });
            return intervals.ToImmutable();
        }
    }
}
