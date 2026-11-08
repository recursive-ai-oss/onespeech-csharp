using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CtcTrellisTests
{
    private static CtcAlignment Align(ReadOnlySpan<float> probabilities, int frames, int vocabulary, ReadOnlySpan<int> target, int blankId = 0) =>
        CtcTrellis.Align(probabilities, frames, vocabulary, target, blankId, TestContext.Current.CancellationToken);

    [Fact]
    public void RepeatedTokensRequireBlankAndKeepSeparateOccupancy()
    {
        float[] emissions = [-9, 0, -9, 0, -9, -9, -9, 0, -9];
        var result = Align(emissions, 3, 3, [1, 1]);
        Assert.True(result.Succeeded);
        Assert.Equal(0, result.LogScore);
        Assert.Equal([new CtcTokenSpan(0, 1, 0, 1), new CtcTokenSpan(1, 1, 2, 3)], result.Tokens);
        var tooShort = Align(new float[6], 2, 3, [1, 1]);
        Assert.False(tooShort.Succeeded);
        Assert.Empty(tooShort.Tokens);
        Assert.Equal(double.NegativeInfinity, tooShort.LogScore);
    }

    [Fact]
    public void BlankRunsAreNotAssignedToTokens()
    {
        float[] emissions = [0, -9, -9, -9, 0, -9, -9, 0, -9, 0, -9, -9, -9, -9, 0, 0, -9, -9];
        var result = Align(emissions, 6, 3, [1, 2]);
        Assert.True(result.Succeeded);
        Assert.Equal([new CtcTokenSpan(0, 1, 1, 3), new CtcTokenSpan(1, 2, 4, 5)], result.Tokens);
        Assert.Equal(0, result.LogScore);
    }

    [Fact]
    public void DistinctTokensCanAdvanceWithoutBlankAndBlankIdIsConfigurable()
    {
        var result = Align([-9, 0, -9, -9, 0, -9, 0, -9, -9], 3, 3, [1, 0], blankId: 2);
        Assert.True(result.Succeeded);
        Assert.Equal([new CtcTokenSpan(0, 1, 0, 2), new CtcTokenSpan(1, 0, 2, 3)], result.Tokens);
    }

    [Fact]
    public void TiesStayAndPreferFinalTokenOverTrailingBlank()
    {
        var result = Align(new float[12], 4, 3, [1, 2]);
        Assert.Equal([new CtcTokenSpan(0, 1, 0, 1), new CtcTokenSpan(1, 2, 1, 4)], result.Tokens);
    }

    [Fact]
    public void ImpossibleEmissionsReturnFailureInsteadOfInventingIntervals()
    {
        var result = Align([0, float.NegativeInfinity, 0, float.NegativeInfinity], 2, 2, [1]);
        Assert.False(result.Succeeded);
        Assert.Empty(result.Tokens);
        Assert.Equal(double.NegativeInfinity, result.LogScore);
    }

    [Fact]
    public void EmptyTargetScoresOnlyBlankAndEmptyAudioIsValid()
    {
        var empty = Align([], 0, 2, []);
        Assert.True(empty.Succeeded);
        Assert.Equal(0, empty.LogScore);
        Assert.Empty(empty.Tokens);
        var blanks = Align([-0.5f, -1, -0.25f, -2], 2, 2, []);
        Assert.True(blanks.Succeeded);
        Assert.Equal(-0.75, blanks.LogScore);
        Assert.Empty(blanks.Tokens);
        Assert.False(Align([], 0, 2, [1]).Succeeded);
        Assert.False(Align([float.NegativeInfinity, 0], 1, 2, []).Succeeded);
    }

    [Fact]
    public void RejectsInvalidShapesIdsProbabilitiesAndExcessiveTrellis()
    {
        Assert.Throws<ArgumentException>(() => Align([], -1, 3, []));
        Assert.Throws<ArgumentException>(() => Align([], 1, 0, []));
        Assert.Throws<ArgumentException>(() => Align([0], 1, 2, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => Align([0], 1, 1, [], blankId: 1));
        Assert.Throws<ArgumentException>(() => Align([0, 0], 1, 2, [0]));
        Assert.Throws<ArgumentException>(() => Align([0, 0], 1, 2, [-1]));
        Assert.Throws<ArgumentException>(() => Align([0, 0], 1, 2, [2]));
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, 0.001f })
            Assert.Throws<ArgumentException>(() => Align([0, invalid], 1, 2, []));
        Assert.Throws<ArgumentException>(() => Align(new float[8002], 4001, 2, new int[2000]));
    }

    [Fact]
    public void CancellationAppliesToEmptyInputToo()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        Assert.Throws<OperationCanceledException>(() => CtcTrellis.Align([], 0, 2, [], cancellationToken: source.Token));
    }

    [Fact]
    public void MatchesExhaustivePathEnumerationOnSmallLattices()
    {
        for (int frames = 0; frames <= 5; frames++)
        for (int variant = 0; variant < 3; variant++)
        {
            float[] emissions = new float[frames * 3];
            for (int i = 0; i < emissions.Length; i++) emissions[i] = -((i * 7 + i / 3 * 11 + variant * 3) % 13) / 4f;
            for (int length = 0; length <= 3; length++)
            for (int encoded = 0; encoded < (1 << length); encoded++)
            {
                int[] target = Enumerable.Range(0, length).Select(i => 1 + ((encoded >> i) & 1)).ToArray();
                var result = Align(emissions, frames, 3, target);
                double best = double.NegativeInfinity;
                int count = (int)Math.Pow(3, frames);
                for (int pathNumber = 0; pathNumber < count; pathNumber++)
                {
                    int number = pathNumber, previous = -1;
                    double score = 0;
                    List<int> collapsed = [];
                    for (int frame = 0; frame < frames; frame++)
                    {
                        int id = number % 3; number /= 3;
                        score += emissions[frame * 3 + id];
                        if (id != 0 && id != previous) collapsed.Add(id);
                        previous = id;
                    }
                    if (collapsed.SequenceEqual(target)) best = Math.Max(best, score);
                }
                Assert.Equal(best, result.LogScore);
                Assert.Equal(double.IsFinite(best), result.Succeeded);
                if (!result.Succeeded) continue;
                Assert.Equal(target.Length, result.Tokens.Length);
                int end = 0;
                for (int i = 0; i < target.Length; i++)
                {
                    var span = result.Tokens[i];
                    Assert.Equal(i, span.TokenIndex);
                    Assert.Equal(target[i], span.TokenId);
                    Assert.InRange(span.StartFrame, end, frames - 1);
                    Assert.InRange(span.EndFrame, span.StartFrame + 1, frames);
                    if (i > 0 && target[i] == target[i - 1]) Assert.True(span.StartFrame > end);
                    end = span.EndFrame;
                }
                // Fill unoccupied frames with blank to independently reconstruct the returned path score.
                double reconstructed = 0;
                for (int frame = 0; frame < frames; frame++)
                {
                    int id = 0;
                    foreach (var span in result.Tokens)
                        if (span.StartFrame <= frame && frame < span.EndFrame) id = span.TokenId;
                    reconstructed += emissions[frame * 3 + id];
                }
                Assert.Equal(best, reconstructed);
            }
        }
    }
}
