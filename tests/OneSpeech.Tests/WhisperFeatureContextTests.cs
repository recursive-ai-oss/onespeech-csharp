using OneSpeech.Audio;
using Xunit;

namespace OneSpeech.Tests;

public sealed class WhisperFeatureContextTests
{
    [Fact]
    public void BoundaryFrameIncludesAudioBeyondTheWindow()
    {
        float[] samples = new float[WhisperLogMelSpectrogram.WindowSamples + 400];
        samples[WhisperLogMelSpectrogram.WindowSamples + 10] = 1;
        var extractor = new WhisperLogMelSpectrogram();
        var audio = new AudioBuffer(samples, 16000);
        float[] context = extractor.ExtractContext(audio, 0, float.NegativeInfinity, TestContext.Current.CancellationToken);
        float[] isolated = extractor.Extract(new AudioBuffer(samples.AsMemory(0, WhisperLogMelSpectrogram.WindowSamples), 16000), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(-1.5f, isolated[^1]);
        Assert.Contains(context.Where((_, index) => index % 3000 == 2999), value => value > -1.5f);
    }

    [Fact]
    public void FileFloorCanOnlyRaiseTheWindowFloor()
    {
        var audio = new AudioBuffer(new float[400], 16000);
        float[] features = new WhisperLogMelSpectrogram().ExtractContext(audio, 0, 0, TestContext.Current.CancellationToken);
        Assert.All(features, value => Assert.Equal(0, value));
    }
}
