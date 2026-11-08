using OneSpeech.Embedders;
using Xunit;

namespace OneSpeech.Tests;

public sealed class SpeakerFeatureTests
{
    [Fact]
    public void SilenceAndSingleFrameNormalizeToZero()
    {
        var extractor = new SpeakerFbank();
        Assert.All(extractor.Compute(new float[16000], CancellationToken.None), x => Assert.Equal(0, x, .0001));
        float[] impulse = new float[400]; impulse[123] = .5f;
        Assert.All(extractor.Compute(impulse, CancellationToken.None), x => Assert.Equal(0, x));
        Assert.Equal(98, SpeakerFbank.FrameCount(16000));
        Assert.Equal(99, SpeakerFbank.FrameCount(16080));
    }
    [Fact]
    public void FeaturesAreFiniteAndMeanNormalized()
    {
        float[] audio = new float[16000];
        for (int i = 0; i < audio.Length; i++) audio[i] = .2f * MathF.Sin(i * .17f) + .1f * MathF.Cos(i * .33f);
        float[] features = new SpeakerFbank().Compute(audio, CancellationToken.None);
        int frames = SpeakerFbank.FrameCount(audio.Length);
        Assert.Equal(80 * frames, features.Length);
        for (int m = 0; m < 80; m++)
        {
            double sum = 0;
            for (int f = 0; f < frames; f++) { Assert.True(float.IsFinite(features[m * frames + f])); sum += features[m * frames + f]; }
            Assert.InRange(Math.Abs(sum / frames), 0, .0001);
        }
    }
    [Fact]
    public void LengthNonfiniteAndCancellationAreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeakerFbank.FrameCount(399));
        Assert.Throws<ArgumentOutOfRangeException>(() => SpeakerFbank.FrameCount(960001));
        var extractor = new SpeakerFbank();
        float[] invalid = new float[400]; invalid[3] = float.NaN;
        Assert.Throws<ArgumentException>(() => extractor.Compute(invalid, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => extractor.Compute(Enumerable.Repeat(float.MaxValue, 400).ToArray(), CancellationToken.None));
        Assert.Throws<OperationCanceledException>(() => extractor.Compute(new float[400], new CancellationToken(true)));
    }
}
