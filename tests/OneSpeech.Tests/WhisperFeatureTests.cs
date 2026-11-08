using System.Text.Json;
using OneSpeech.Audio;
using Xunit;

namespace OneSpeech.Tests;

public sealed class WhisperFeatureTests
{
    [Theory]
    [InlineData(80)]
    [InlineData(128)]
    public void SilenceIsMinusOnePointFiveAndHasModelShape(int bins)
    {
        float[] features = new WhisperLogMelSpectrogram(bins).Extract(new AudioBuffer(ReadOnlyMemory<float>.Empty, 16000), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(bins * 3000, features.Length);
        Assert.All(features, value => Assert.Equal(-1.5f, value));
    }

    [Theory]
    [InlineData(80)]
    [InlineData(128)]
    public void ToneMatchesBlackBoxObservations(int bins)
    {
        float[] samples = Enumerable.Range(0, 16000).Select(i => (float)(.25 * Math.Sin(2 * Math.PI * 440 * i / 16000))).ToArray();
        float[] features = new WhisperLogMelSpectrogram(bins).Extract(new AudioBuffer(samples, 16000), cancellationToken: TestContext.Current.CancellationToken);
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, $"Fixtures/tone-{bins}.json")));
        int[] indices = reference.RootElement.GetProperty("indices").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        float[] values = reference.RootElement.GetProperty("values").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        for (int i = 0; i < indices.Length; i++) Assert.InRange(Math.Abs(features[indices[i]] - values[i]), 0, 3e-5f);
    }

    [Fact]
    public void ValidatesInputAndCancellation()
    {
        var extractor = new WhisperLogMelSpectrogram();
        Assert.Throws<ArgumentOutOfRangeException>(() => new WhisperLogMelSpectrogram(64));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new float[480001], 16000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new float[400], 8000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new float[400], 16000, 2), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new[] { float.NaN }, 16000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new[] { float.MaxValue }, 16000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new float[200], 16000), padToWindow: false, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<OperationCanceledException>(() => extractor.Extract(new AudioBuffer(new float[400], 16000), cancellationToken: new CancellationToken(true)));
        Assert.Equal(80 * 2, extractor.Extract(new AudioBuffer(new float[400], 16000), padToWindow: false, cancellationToken: TestContext.Current.CancellationToken).Length);
    }
}
