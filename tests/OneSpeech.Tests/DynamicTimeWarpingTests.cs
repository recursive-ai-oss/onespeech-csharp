using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class DynamicTimeWarpingTests
{
    [Fact]
    public void FollowsKnownMinimumAndPrefersDiagonalTies()
    {
        var path = DynamicTimeWarping.Align(new float[] { 0, 9, 9, 9, 0, 9, 9, 9, 0 }, 3, 3, TestContext.Current.CancellationToken);
        Assert.Equal(new[] { (0, 0), (1, 1), (2, 2) }, path);
        Assert.Equal(new[] { (0, 0), (1, 1) }, DynamicTimeWarping.Align(new float[4], 2, 2, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void HandlesRepeatedFramesAndChecksBounds()
    {
        Assert.Equal(new[] { (0, 0), (0, 1), (0, 2) }, DynamicTimeWarping.Align(new float[3], 1, 3, TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => DynamicTimeWarping.Align(new float[3], 2, 2, TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => DynamicTimeWarping.Align(new[] { float.NaN }, 1, 1, TestContext.Current.CancellationToken));
        Assert.Throws<OperationCanceledException>(() => DynamicTimeWarping.Align(new float[1], 1, 1, new CancellationToken(true)));
    }
}
