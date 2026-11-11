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
    [Theory]
    [InlineData(400)]
    [InlineData(16000)]
    [InlineData(16081)]
    [InlineData(48000)]
    public void SparseFilterAndCachedTwiddlesMatchDenseArithmetic(int length)
    {
        var random = new Random(1729);
        var samples = new float[length];
        for (int i = 0; i < length; i++) samples[i] = (float)(random.NextDouble() - .5);
        Assert.Equal(new DenseSpeakerFbank().Compute(samples, CancellationToken.None), new SpeakerFbank().Compute(samples, CancellationToken.None));
    }

    [Fact]
    public void FivePublicSpeechProbesPreserveFeatures()
    {
        string? corpus = Environment.GetEnvironmentVariable("ONESPEECH_PROBE_CORPUS");
        Assert.SkipWhen(corpus is null, "Set ONESPEECH_PROBE_CORPUS to the public LibriSpeech corpus root.");
        string? reference = Environment.GetEnvironmentVariable("ONESPEECH_SPEAKER_REFERENCE");
        double maximum = 0;
        var baseline = new DenseSpeakerFbank();
        var optimized = new SpeakerFbank();
        for (int i = 0; i < 5; i++)
        {
            string id = $"1089-134686-{i:D4}";
            var audio = OneAudio.Audio.Load(Path.Combine(corpus!, "test-clean", "1089", "134686", id + ".flac"), cancellationToken: TestContext.Current.CancellationToken);
            var before = baseline.Compute(audio.Samples.Span, CancellationToken.None);
            var after = optimized.Compute(audio.Samples.Span, CancellationToken.None);
            Assert.Equal(before, after);
            if (reference is null) continue;
            byte[] bytes = File.ReadAllBytes(Path.Combine(reference, id + ".features.f32"));
            Assert.Equal(after.Length * 4, bytes.Length);
            for (int j = 0; j < after.Length; j++)
                maximum = Math.Max(maximum, Math.Abs((double)after[j] - System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(j * 4))));
        }
        TestContext.Current.TestOutputHelper!.WriteLine($"CAM++ five-probe maximum absolute reference error: {maximum:R}; baseline/optimized mismatch count: 0.");
        Assert.InRange(maximum, 0, .001);
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
        Assert.Throws<ArgumentException>(() => extractor.Compute(Enumerable.Range(0, 400).Select(i => i % 2 == 0 ? 1e19f : -1e19f).ToArray(), CancellationToken.None));
        Assert.Throws<OperationCanceledException>(() => extractor.Compute(new float[400], new CancellationToken(true)));
    }
}
