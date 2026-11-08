using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class BeamSearchTests
{
    private static readonly int[] Better = [2];
    private static readonly int[] Greedy = [1];
    private static readonly int[] Tied = [1, 2];
    private sealed class State(float[] logits, Func<int, State> advance) : IBeamDecoderState
    {
        public ReadOnlyMemory<float> Logits => logits;
        public IBeamDecoderState Advance(int token, CancellationToken cancellationToken) { cancellationToken.ThrowIfCancellationRequested(); return advance(token); }
        public void Dispose() { }
    }
    private static State Lattice() => new([float.NegativeInfinity, MathF.Log(.55f), MathF.Log(.45f)], token => new State(token == 1 ? [MathF.Log(.5f), MathF.Log(.25f), MathF.Log(.25f)] : [MathF.Log(.99f), MathF.Log(.005f), MathF.Log(.005f)], _ => throw new InvalidOperationException("No further advance expected.")));

    [Fact]
    public void BeamFindsTheKnownBetterSequenceWhileWidthOneIsGreedy()
    {
        Assert.Equal(Better, BeamSearch.Decode(Lattice(), 0, beamSize: 2, cancellationToken: TestContext.Current.CancellationToken).Tokens);
        Assert.Equal(Greedy, BeamSearch.Decode(Lattice(), 0, beamSize: 1, cancellationToken: TestContext.Current.CancellationToken).Tokens);
    }
    [Fact]
    public void SelectionTiesMasksAndInvalidValuesAreExplicit()
    {
        Assert.Equal(Tied, LogitMath.TopIndices(new float[] { -1, 0, 0 }, 2));
        Assert.Equal(0, LogitMath.ArgMax(new float[] { 1, 1 }));
        Assert.Equal(double.NegativeInfinity, LogitMath.LogSumExp(new float[] { float.NegativeInfinity }));
        Assert.Throws<ArgumentException>(() => LogitMath.ArgMax(new float[] { float.NaN }));
        Assert.Throws<OperationCanceledException>(() => BeamSearch.Decode(Lattice(), 0, cancellationToken: new CancellationToken(true)));
    }
}
