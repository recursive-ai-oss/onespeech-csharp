using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ContextualCtcTests
{
    [Fact]
    public void EmptyContextMatchesExistingDecoder()
    {
        float[] logits = [-.3f, -.5f, -1, -.8f, -.2f, -.3f, -.5f, -.4f, -.8f];
        var baseline = CtcPrefixBeamSearch.Decode(logits, 3, 2, beamSize: 64, cancellationToken: TestContext.Current.CancellationToken);
        var contextual = ContextualCtcBeamSearch.Decode(logits, 3, 2, new PhraseTrie([]), beamSize: 64, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(baseline.Tokens, contextual.Tokens);
        Assert.Equal(baseline.LogProbability, contextual.LogProbability);
        Assert.Equal(0, contextual.ContextScore);
    }

    [Fact]
    public void ScoresEveryCollapsedPrefixOnceNotEveryFrame()
    {
        float[] probabilities = [.45f, .35f, .2f, .3f, .2f, .5f, .2f, .5f, .3f];
        var paths = new Dictionary<string, (int[] Tokens, double Probability)>(StringComparer.Ordinal);
        var trie = new PhraseTrie([new([0, 1], .8), new([0], .2)]);
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 3; c++)
                {
                    List<int> ids = [];
                    int previous = 2;
                    foreach (int id in new[] { a, b, c }) { if (id != previous && id != 2) ids.Add(id); previous = id; }
                    string key = string.Join(',', ids);
                    paths.TryGetValue(key, out var old);
                    paths[key] = (ids.ToArray(), old.Probability + (double)probabilities[a] * probabilities[3 + b] * probabilities[6 + c]);
                }
        var expected = paths.Values.MaxBy(path => Math.Log(path.Probability) + trie.Score(path.Tokens));
        var result = ContextualCtcBeamSearch.Decode(probabilities.Select(value => MathF.Log(value)).ToArray(), 3, 2, trie, beamSize: 64, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(expected.Tokens, result.Tokens);
        Assert.Equal(Math.Log(expected.Probability), result.LogProbability, 6);
        Assert.Equal(trie.Score(expected.Tokens), result.ContextScore, 6);
    }

    [Fact]
    public void CompletedPhraseCanChangeWinnerButUnfinishedCreditIsRemoved()
    {
        float[] logits = [0, -.5f, -10];
        var result = ContextualCtcBeamSearch.Decode(logits, 3, 2, new PhraseTrie([new([1], 1)]), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([1], result.Tokens);
        Assert.Equal(1, result.ContextScore);
        var unfinished = ContextualCtcBeamSearch.Decode(logits, 3, 2, new PhraseTrie([new([1, 0], 10)]), beamSize: 1, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0], unfinished.Tokens);
        Assert.Equal(0, unfinished.ContextScore);
    }

    [Fact]
    public void BlankPhrasesAndCancellationAreRefused()
    {
        Assert.Throws<ArgumentException>(() => ContextualCtcBeamSearch.Decode([0, 0], 2, 1, new PhraseTrie([new([1])]), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => ContextualCtcBeamSearch.Decode([0, 0], 2, 1, new PhraseTrie([new([2])]), cancellationToken: TestContext.Current.CancellationToken));
        using var source = new CancellationTokenSource(); source.Cancel();
        Assert.Throws<OperationCanceledException>(() => ContextualCtcBeamSearch.Decode([], 2, 1, new PhraseTrie([]), cancellationToken: source.Token));
    }
}
