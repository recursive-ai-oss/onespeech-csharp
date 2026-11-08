using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class PhraseTrieTests
{
    [Fact]
    public void PartialCreditIsRepaidOnMismatchAndFinalization()
    {
        var trie = new PhraseTrie([new([1, 2, 3], 2)]);
        Assert.Equal(2, trie.Advance(0, 1, out int state));
        Assert.Equal(-2, trie.Finish(state));
        Assert.Equal(-2, trie.Advance(state, 9, out int reset));
        Assert.Equal(0, reset);
        Assert.Equal(0, trie.Score([1, 2, 9]));
        Assert.Equal(0, trie.Score([1, 2]));
        Assert.Equal(6, trie.Score([1, 2, 3]));
        Assert.Equal(12, trie.Score([1, 2, 3, 1, 2, 3]));
    }

    [Fact]
    public void FailureLinksKeepOverlappingPrefixesAndUseLargestCompletion()
    {
        TokenPhrase[] phrases = [new([1, 2], 2), new([2, 1], 3), new([1, 2], 1), new([2], 1)];
        var trie = new PhraseTrie(phrases);
        for (int bits = 0; bits < 729; bits++)
        {
            int value = bits;
            int[] tokens = new int[6];
            for (int i = 0; i < tokens.Length; i++) { tokens[i] = value % 3; value /= 3; }
            double expected = 0;
            for (int end = 1; end <= tokens.Length; end++)
            {
                double largest = 0;
                foreach (var phrase in phrases)
                    if (phrase.Tokens.Length <= end && tokens.AsSpan(end - phrase.Tokens.Length, phrase.Tokens.Length).SequenceEqual(phrase.Tokens.AsSpan()))
                        largest = Math.Max(largest, phrase.Tokens.Length * phrase.Weight);
                expected += largest;
            }
            Assert.Equal(expected, trie.Score(tokens));
        }
    }

    [Fact]
    public void EmptyTrieIsNeutralAndHostileBoundsAreRejected()
    {
        var empty = new PhraseTrie([]);
        Assert.Equal(0, empty.Score([1, 2, 3]));
        Assert.Equal(0, empty.Score([]));
        Assert.Throws<ArgumentException>(() => new PhraseTrie([new([])]));
        Assert.Throws<ArgumentException>(() => new PhraseTrie([new([-1])]));
        Assert.Throws<ArgumentException>(() => new PhraseTrie([new([1], double.NaN)]));
        Assert.Throws<ArgumentException>(() => new PhraseTrie([new([1], 0)]));
        Assert.Throws<ArgumentException>(() => new PhraseTrie(Enumerable.Repeat(new TokenPhrase([1]), 4097)));
        Assert.Throws<ArgumentOutOfRangeException>(() => empty.Finish(1));
    }
}
