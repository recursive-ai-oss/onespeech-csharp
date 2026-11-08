using OneSpeech.Diarization;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ShortClusteringTests
{
    // Numerical geometry probes, not voice embeddings or synthesized speakers.
    private static EmbeddingWindow Window(int i, float x, float y, bool tail = false)
    {
        float[] vector = [x, y]; EmbeddingClustering.Normalize(vector);
        return new EmbeddingWindow
        {
            Start = TimeSpan.FromSeconds(i * 0.25),
            End = TimeSpan.FromSeconds(i * 0.25 + (tail ? 0.1 : 1.5)),
            Embedding = vector
        };
    }

    [Fact]
    public void CompactCloudDoesNotBecomeNearestNeighborComponents()
    {
        var windows = Enumerable.Range(0, 32).Select(i => Window(i, 1, i * 0.005f)).ToArray();
        Assert.Single(EmbeddingClustering.Cluster(windows, new(), TestContext.Current.CancellationToken).Distinct());
    }

    [Fact]
    public void TruncatedTailCannotIntroduceAnotherCluster()
    {
        var windows = Enumerable.Range(0, 32).Select(i => Window(i, 1, 0)).Append(Window(32, 0, 1, true)).ToArray();
        Assert.Single(EmbeddingClustering.Cluster(windows, new(), TestContext.Current.CancellationToken).Distinct());
    }

    [Theory]
    [InlineData(null, 2)]
    [InlineData(1, 1)]
    [InlineData(4, 4)]
    public void DistinctCloudsAndExactCountsArePreserved(int? hint, int expected)
    {
        var windows = Enumerable.Range(0, 32).Select(i => i < 16 ? Window(i, 1, 0) : Window(i, 0, 1)).ToArray();
        var options = new ClusteringOptions { NumSpeakers = hint };
        Assert.Equal(expected, EmbeddingClustering.Cluster(windows, options, TestContext.Current.CancellationToken).Distinct().Count());
    }

    [Fact]
    public void SubWindowAudioAndMinimumCountRemainSupported()
    {
        EmbeddingWindow[] windows = [Window(0, 1, 0, true), Window(1, 0, 1, true)];
        Assert.Equal(2, EmbeddingClustering.Cluster(windows, new() { MinSpeakers = 2 }, TestContext.Current.CancellationToken).Distinct().Count());
        Assert.Throws<ArgumentException>(() => new ClusteringOptions { ShortRecordingThreshold = double.NaN }.Validate());
        Assert.Throws<OperationCanceledException>(() => EmbeddingClustering.Cluster(windows, new(), new CancellationToken(true)));
    }
}
