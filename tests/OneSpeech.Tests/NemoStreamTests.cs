using OneSpeech.Audio;
using Xunit;

namespace OneSpeech.Tests;

public sealed class NemoStreamTests
{
    [Fact]
    public void FrameReadinessUsesHannSupportRatherThanFftPadding()
    {
        var state = new NemoMelStream(new NemoLogMelSpectrogram()); int count = 0;
        var ct = TestContext.Current.CancellationToken;
        state.Push(new float[199], false, _ => count++, ct); Assert.Equal(0, count);
        state.Push([0], false, _ => count++, ct); Assert.Equal(1, count);
        state.Push(new float[160], false, _ => count++, ct); Assert.Equal(2, count);
    }

    [Theory]
    [InlineData(320, 1)]
    [InlineData(512, 83)]
    [InlineData(16000, 1280)]
    [InlineData(16117, 4096)]
    public void StreamingRawFeaturesEqualBatchIncludingPaddedTail(int samples, int chunk)
    {
        var ct = TestContext.Current.CancellationToken;
        float[] pcm = Enumerable.Range(0, samples).Select(i => .2f * MathF.Sin(i * .07f) + .1f * MathF.Sin(i * .013f)).ToArray();
        var front = new NemoLogMelSpectrogram();
        var state = new NemoMelStream(front);
        float[] expected = front.Extract(new AudioBuffer(pcm, 16000), normalize: false, cancellationToken: ct);
        var output = new List<float[]>();
        for (int start = 0; start < samples; start += chunk) state.Push(pcm.AsSpan(start, Math.Min(chunk, samples - start)), false, values => output.Add((float[])values.Clone()), ct);
        state.Push([], true, values => output.Add((float[])values.Clone()), ct);
        int frames = samples / 160;
        Assert.Equal(frames, output.Count); Assert.Equal(frames, state.FrameCount);
        for (int channel = 0; channel < 128; channel++) for (int frame = 0; frame < frames; frame++) Assert.Equal(expected[channel * (frames + 1) + frame], output[frame][channel]);
        Assert.Throws<InvalidOperationException>(() => state.Push([], false, _ => { }, ct));
        state.Reset(); Assert.Equal(0, state.FrameCount);
        var reset = new List<float[]>(); state.Push(pcm, true, values => reset.Add((float[])values.Clone()), ct);
        Assert.Equal(output.Count, reset.Count);
        for (int frame = 0; frame < frames; frame++) Assert.Equal(output[frame], reset[frame]);
    }
}
