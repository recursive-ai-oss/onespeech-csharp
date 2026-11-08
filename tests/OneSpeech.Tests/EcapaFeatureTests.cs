using OneSpeech.Embedders;
using Xunit;

namespace OneSpeech.Tests;

public sealed class EcapaFeatureTests
{
    [Fact]
    public void CenteredSilenceAndFrameBounds()
    {
        var extractor = new EcapaFbank();
        Assert.All(extractor.ComputeRaw(new float[16000], CancellationToken.None), value => Assert.Equal(-100, value));
        Assert.All(extractor.Compute(new float[16000], CancellationToken.None), value => Assert.Equal(0, value));
        Assert.Equal(101, EcapaFbank.FrameCount(16000));
        Assert.Equal(101, EcapaFbank.FrameCount(16001));
        Assert.Equal(1, EcapaFbank.FrameCount(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EcapaFbank.FrameCount(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => EcapaFbank.FrameCount(960001));
    }
    [Fact]
    public void MixedRadixSpectrumMatchesPublicToneProbe()
    {
        float[] audio = new float[16001];
        for (int i = 0; i < audio.Length; i++) audio[i] = MathF.Sin(i * (2 * MathF.PI * 440 / 16000)) * .1f;
        var features = new EcapaFbank().ComputeRaw(audio, CancellationToken.None);
        int frames = EcapaFbank.FrameCount(audio.Length);
        // Numeric output of the pinned released oracle on the generated tone, not source fixtures.
        Assert.Equal(-12.52259445, features[0], .001);
        Assert.Equal(-6.40787315, features[frames], .001);
        Assert.Equal(-6.72996426, features[2 * frames], .001);
        Assert.InRange(features.Max() - features.Min(), 79.9999, 80.0001);
    }
    [Fact]
    public void InvalidMagnitudeAndCancellationAreRefused()
    {
        var features = new EcapaFbank();
        Assert.Throws<ArgumentException>(() => features.Compute(new float[] { float.NaN }, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => features.Compute(Enumerable.Repeat(float.MaxValue, 1000).ToArray(), CancellationToken.None));
        Assert.Throws<OperationCanceledException>(() => features.Compute(new float[400], new CancellationToken(true)));
    }
}
