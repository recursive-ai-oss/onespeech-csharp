using System.Collections.Immutable;
using OneSpeech.Diarization;
using Xunit;

namespace OneSpeech.Tests;

public sealed class SegmentationSlotTests
{
    private static readonly ModelInfo Provenance = new() { Id = "segmentation-fixture", Revision = "1", Source = "synthetic", License = "MIT", Attribution = "Synthetic interval fixture", Tier = ModelTier.Tier1 };
    [Fact]
    public void SegmentationRoleReplacesActivityDetectionWithoutFillingGaps()
    {
        using var embedding = new ConstantEmbedder(); var segmentation = new SegmentationFixture();
        var engine = new ClusteringDiarizer(embedding, segmentation, new() { NumSpeakers = 1, WindowSeconds = .5, ShiftSeconds = .5 });
        var turns = engine.Diarize(new(new float[4 * 16000], 16000), TestContext.Current.CancellationToken);
        Assert.Equal(2, turns.Length);
        Assert.Equal(TimeSpan.FromSeconds(.5), turns[0].Start); Assert.Equal(TimeSpan.FromSeconds(1.5), turns[0].End);
        Assert.Equal(TimeSpan.FromSeconds(2.5), turns[1].Start); Assert.Equal(TimeSpan.FromSeconds(3.5), turns[1].End);
        Assert.Equal(1, segmentation.Calls); Assert.Equal(4, embedding.Calls);
        Assert.Contains(engine.Models, m => m.Id == Provenance.Id);
    }
    private sealed class SegmentationFixture : VoiceActivityDetector
    {
        internal int Calls;
        public override ModelInfo Model => Provenance;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); Calls++;
            return [new() { Start = TimeSpan.FromSeconds(.5), End = TimeSpan.FromSeconds(1.5) }, new() { Start = TimeSpan.FromSeconds(2.5), End = TimeSpan.FromSeconds(3.5) }];
        }
    }
    private sealed class ConstantEmbedder : SpeakerEmbedder
    {
        internal int Calls;
        public override ModelInfo Model => Provenance with { Id = "embedding-fixture" };
        public override int SampleRate => 16000;
        public override int Dimension => 2;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); Calls++; Assert.Equal(8000, audio.FrameCount); destination[0] = 1; destination[1] = 0;
        }
        public override void Dispose() { }
    }
}
