using Xunit;
using OneSpeech.Audio;
using OneSpeech.Decoding;

namespace OneSpeech.Tests;

public sealed class TransducerTests
{
    [Fact]
    public void CtcCollapseResetsAtBlankAndKeepsRunIntervals()
    {
        float[] logits = [9, 0, 0, 8, 0, 0, 0, 0, 9, 9, 0, 0, 0, 9, 0];
        var tokens = CtcGreedy.Decode(logits, 3, 2, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0, 0, 1], tokens.Select(t => t.Id));
        Assert.Equal([(0, 2), (3, 4), (4, 5)], tokens.Select(t => (t.StartFrame, t.EndFrame)));
        Assert.All(tokens, t => Assert.InRange(t.Confidence, .99f, 1));
        Assert.Empty(CtcGreedy.Decode([], 3, 2, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CtcBeamSumsPathsThatGreedyDoesNot()
    {
        // Two rows: a=.4, blank=.6. Empty has .36; collapsed a has .16+.24+.24=.64.
        float[] row = [MathF.Log(.4f), MathF.Log(.6f)];
        Assert.Empty(CtcGreedy.Decode([.. row, .. row], 2, 1, cancellationToken: TestContext.Current.CancellationToken));
        var result = CtcPrefixBeamSearch.Decode([.. row, .. row], 2, 1, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0], result.Tokens);
        Assert.Equal(Math.Log(.64), result.LogProbability, 6);
        Assert.Equal([0, 0], CtcPrefixBeamSearch.Decode([0, float.NegativeInfinity, float.NegativeInfinity, 0, 0, float.NegativeInfinity], 2, 1, cancellationToken: TestContext.Current.CancellationToken).Tokens);
    }

    [Fact]
    public void CtcPrefixScoresMatchExhaustivePaths()
    {
        float[] probabilities = [.45f, .35f, .2f, .3f, .2f, .5f, .2f, .5f, .3f];
        var expected = new Dictionary<string, double>(StringComparer.Ordinal);
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 3; c++)
                {
                    List<int> ids = [];
                    int previous = 2;
                    foreach (int id in new[] { a, b, c }) { if (id != previous && id != 2) ids.Add(id); previous = id; }
                    string key = string.Join(',', ids);
                    expected.TryGetValue(key, out double sum);
                    expected[key] = sum + (double)probabilities[a] * probabilities[3 + b] * probabilities[6 + c];
                }
        var best = expected.MaxBy(kv => kv.Value);
        var actual = CtcPrefixBeamSearch.Decode(probabilities.Select(value => MathF.Log(value)).ToArray(), 3, 2, beamSize: 64, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(best.Key, string.Join(',', actual.Tokens));
        Assert.Equal(Math.Log(best.Value), actual.LogProbability, 6);
    }

    [Fact]
    public void RnntNonblankDoesNotAdvanceOrCollapseRepeats()
    {
        List<(int Frame, int Prefix)> calls = [];
        void Joint(int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> duration, CancellationToken ct)
        {
            calls.Add((frame, prefix.Length));
            logits.Fill(-10);
            logits[prefix.Length < 2 ? 0 : 1] = 10;
            Assert.Empty(duration.ToArray());
        }
        var result = RnntGreedy.Decode(2, 2, 1, Joint, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0, 0], result.Select(t => t.Id));
        Assert.Equal([(0, 0), (0, 1), (0, 2), (1, 2)], calls);
        Assert.All(result, token => { Assert.Equal(0, token.StartFrame); Assert.Equal(1, token.EndFrame); });
    }

    [Fact]
    public void TdtSkipsFramesAndPreservesZeroDurations()
    {
        List<int> calledFrames = [];
        void Joint(int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> duration, CancellationToken ct)
        {
            calledFrames.Add(frame);
            logits.Fill(-10); duration.Fill(-10);
            if (prefix.Length == 0) { logits[0] = 10; duration[0] = 10; }
            else if (prefix.Length == 1) { logits[1] = 10; duration[2] = 10; }
            else { logits[2] = 10; duration[0] = 10; } // Blank + duration 0 must progress.
        }
        var result = TdtGreedy.Decode(4, 3, 2, [0, 1, 3], Joint, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0, 0, 3], calledFrames);
        Assert.Equal([(0, 0), (0, 3)], result.Select(t => (t.StartFrame, t.EndFrame)));
        Assert.Equal([0, 1], result.Select(t => t.Id));
    }

    [Fact]
    public void BoundsAndBadLogitsAreNotSilentlyAccepted()
    {
        static void Stuck(int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> duration, CancellationToken ct)
        { logits[0] = 1; logits[1] = 0; duration.Clear(); }
        Assert.Throws<InvalidOperationException>(() => RnntGreedy.Decode(1, 2, 1, Stuck, maxSymbolsPerFrame: 2, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => TdtGreedy.Decode(1, 2, 1, [0, 1], Stuck, maxTokens: 2, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => CtcGreedy.Decode([float.NaN, 0], 2, 1, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => CtcGreedy.Decode([float.NegativeInfinity, float.NegativeInfinity], 2, 1, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => CtcGreedy.Decode([float.PositiveInfinity, 0], 2, 1, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => CtcGreedy.Decode([0], 2, 1, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => TdtGreedy.Decode(1, 2, 1, [0, 0], Stuck, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<OperationCanceledException>(() => RnntGreedy.Decode(1, 2, 1, Stuck, cancellationToken: new(true)));
        Assert.Equal(0, CtcGreedy.Decode([1, 1], 2, 1, cancellationToken: TestContext.Current.CancellationToken)[0].Id);
    }

    [Fact]
    public void NativeWordsUseOnlyTheirTokensAndDoNotFillGaps()
    {
        FrameToken[] tokens = [new(7, 2, 2, .8f), new(8, 2, 4, .6f), new(9, 7, 8, 1)];
        var words = NativeWordTiming.Map(tokens, [new("hello", 0, 2), new("!", 2, 0), new("world", 2, 1)], TimeSpan.FromMilliseconds(80), TimeSpan.FromSeconds(1), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TimeSpan.FromMilliseconds(1160), words[0].Start);
        Assert.Equal(TimeSpan.FromMilliseconds(1320), words[0].End);
        Assert.Equal(.7, words[0].Confidence!.Value, 6);
        Assert.Equal(TimingMethod.Native, words[0].TimingMethod);
        Assert.Null(words[1].Start); Assert.Equal(TimingMethod.None, words[1].TimingMethod);
        Assert.Equal(TimeSpan.FromMilliseconds(1560), words[2].Start);
        Assert.Null(words[0].Speaker);
        Assert.Throws<ArgumentException>(() => NativeWordTiming.Map(tokens, [new("bad", -1, 2)], TimeSpan.FromMilliseconds(80), cancellationToken: TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData(320)]
    [InlineData(400)]
    [InlineData(512)]
    [InlineData(16000)]
    public void NemoSilenceIsFiniteNormalizedAndTimePaddingIsZero(int length)
    {
        var extractor = new NemoLogMelSpectrogram();
        var result = extractor.Extract(new AudioBuffer(new float[length], 16000), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(128 * (NemoLogMelSpectrogram.GetFrameCount(length) + 1), result.Length);
        Assert.All(result, value => Assert.True(float.IsFinite(value)));
        int stride = result.Length / 128, frames = NemoLogMelSpectrogram.GetFrameCount(length);
        for (int mel = 0; mel < 128; mel++) Assert.Equal(0, result[mel * stride + frames]);
    }

    [Fact]
    public void NemoRejectsUnsupportedPcmAndHonorsCancellation()
    {
        var extractor = new NemoLogMelSpectrogram();
        Assert.Throws<ArgumentOutOfRangeException>(() => extractor.Extract(new AudioBuffer(new float[256], 16000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(new float[1000], 8000), cancellationToken: TestContext.Current.CancellationToken));
        float[] bad = new float[1000]; bad[19] = float.NaN;
        Assert.Throws<ArgumentException>(() => extractor.Extract(new AudioBuffer(bad, 16000), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<OperationCanceledException>(() => extractor.Extract(new AudioBuffer(new float[1000], 16000), cancellationToken: new(true)));
    }
}
