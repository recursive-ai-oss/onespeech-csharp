using OneSpeech.Diarization;
using Xunit;

namespace OneSpeech.Tests;

// Algebraic unit vectors test path costs; these are not voice-derived biometric embeddings.
public sealed class CosineResegmentationTests
{
    private static readonly int[] Corrected = [3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9];
    private static readonly int[] Separated = [0, 0, 1, 1];
    private static readonly int[] Singleton = [7];
    private static EmbeddingWindow W(int i, bool second, int gap = 0) => new()
    {
        Start = TimeSpan.FromSeconds(i + gap), End = TimeSpan.FromSeconds(i + gap + 1),
        Embedding = second ? new float[] { 0, 1 } : new float[] { 1, 0 }
    };
    [Fact]
    public void StableCentroidsCorrectAnIsolatedClusteringLabel()
    {
        var windows = Enumerable.Range(0, 12).Select(i => W(i, i >= 6)).ToArray();
        int[] labels = [3, 3, 9, 3, 3, 3, 9, 9, 9, 9, 9, 9];
        var result = new CosineResegmenter(0.1, 1).Resegment(windows, labels, CancellationToken.None);
        Assert.Equal(Corrected, result);
        Assert.Equal(9, labels[2]); Assert.Equal(new float[] { 1, 0 }, windows[0].Embedding.ToArray());
    }
    [Fact]
    public void SwitchPenaltyRemovesShortExcursionsButResetsAtActivityGaps()
    {
        var windows = Enumerable.Range(0, 12).Select(i => W(i, i == 3 || i >= 6)).ToArray();
        int[] labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        Assert.Equal(1, new CosineResegmenter(0, 1).Resegment(windows, labels, CancellationToken.None)[3]);
        Assert.Equal(0, new CosineResegmenter(1, 1).Resegment(windows, labels, CancellationToken.None)[3]);
        var separated = new[] { W(0, false), W(1, false), W(2, true, 4), W(3, true, 4) };
        Assert.Equal(Separated, new CosineResegmenter(10, 0).Resegment(separated, [0, 0, 1, 1], CancellationToken.None));
    }
    [Fact]
    public void FileHelperUsesTheDevelopmentSelectedProfile()
    {
        Assert.Equal(0.4, OneSpeech.Pipeline.SpeechServices.DiarizationResegmentation.SwitchPenalty);
        Assert.Equal(2, OneSpeech.Pipeline.SpeechServices.DiarizationResegmentation.StableRadius);
    }
    [Fact]
    public void EmptySingletonCancellationAndMalformedInputs()
    {
        var smoother = new CosineResegmenter();
        Assert.Empty(smoother.Resegment([], [], CancellationToken.None));
        Assert.Equal(Singleton, smoother.Resegment([W(0, false)], [7], CancellationToken.None));
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => smoother.Resegment([], [], canceled.Token));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineResegmenter(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineResegmenter(0, 17));
        Assert.Throws<ArgumentException>(() => smoother.Resegment([W(0, false)], [], CancellationToken.None));
        Assert.Throws<ArgumentException>(() => smoother.Resegment([W(1, false), W(0, true)], [0, 1], CancellationToken.None));
        Assert.Throws<ArgumentException>(() => smoother.Resegment([W(0, false) with { Embedding = new float[] { float.NaN, 0 } }], [0], CancellationToken.None));
        Assert.Throws<ArgumentException>(() => smoother.Resegment([W(0, false)], [-1], CancellationToken.None));
    }
}
