using OneSpeech.Decoding;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ContextualTransducerTests
{
    private static void Joint(int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> durations, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        logits.Fill(-10);
        if (prefix.Length <= frame) { logits[0] = 0; logits[1] = -.5f; }
        else logits[2] = 0;
        if (!durations.IsEmpty) { durations[0] = 0; durations[1] = prefix.Length <= frame ? -10 : 10; }
    }

    [Fact]
    public void RnntBiasChangesChoiceAndPreservesAcousticConfidence()
    {
        var baseline = RnntGreedy.Decode(2, 3, 2, Joint, cancellationToken: TestContext.Current.CancellationToken);
        var biased = ContextualTransducerSearch.DecodeRnnt(2, 3, 2, Joint, new PhraseTrie([new([1], 1)]), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([0, 0], baseline.Select(t => t.Id));
        Assert.Equal([1, 1], biased.Select(t => t.Id));
        Assert.All(biased, token => Assert.Equal((float)(Math.Exp(-.5) / (1 + Math.Exp(-.5) + Math.Exp(-10))), token.Confidence, 6));
        Assert.Equal([(0, 1), (1, 2)], biased.Select(t => (t.StartFrame, t.EndFrame)));
    }

    [Fact]
    public void EmptyContextIsExactlyNeutralForBothTransducers()
    {
        var trie = new PhraseTrie([]);
        Assert.Equal(RnntGreedy.Decode(3, 3, 2, Joint, cancellationToken: TestContext.Current.CancellationToken),
            ContextualTransducerSearch.DecodeRnnt(3, 3, 2, Joint, trie, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(TdtGreedy.Decode(3, 3, 2, [0, 1], Joint, cancellationToken: TestContext.Current.CancellationToken),
            ContextualTransducerSearch.DecodeTdt(3, 3, 2, [0, 1], Joint, trie, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void TdtKeepsDurationHeadAndSymbolBounds()
    {
        var biased = ContextualTransducerSearch.DecodeTdt(2, 3, 2, [0, 1], Joint, new PhraseTrie([new([1], 1)]), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([1, 1], biased.Select(t => t.Id));
        Assert.Equal([(0, 0), (1, 1)], biased.Select(t => (t.StartFrame, t.EndFrame)));
        void Stuck(int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> durations, CancellationToken ct)
        { ct.ThrowIfCancellationRequested(); logits[0] = 0; logits[1] = -10; durations[0] = 0; durations[1] = -10; }
        Assert.Throws<InvalidOperationException>(() => ContextualTransducerSearch.DecodeTdt(1, 2, 1, [0, 1], Stuck,
            new PhraseTrie([new([0], 1)]), maxSymbolsPerFrame: 2, cancellationToken: TestContext.Current.CancellationToken));
    }
}
