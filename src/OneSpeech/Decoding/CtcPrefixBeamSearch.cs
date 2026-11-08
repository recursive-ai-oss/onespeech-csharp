using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A CTC collapsed sequence and its summed natural log probability over retained paths.</summary>
/// <param name="Tokens">Nonblank token IDs.</param>
/// <param name="LogProbability">Natural log probability; pruning can discard contributing paths.</param>
public sealed record CtcBeamResult(ImmutableArray<int> Tokens, double LogProbability);

/// <summary>CTC prefix beam search, summing blank/nonblank path probabilities separately before pruning.</summary>
public static class CtcPrefixBeamSearch
{
    private static readonly PhraseTrie EmptyContext = new([]);

    /// <summary>Decodes frame-major logits with no language model, bias or length normalization. Ties use lexicographic token order.
    /// Empty input returns the empty prefix with log probability zero. This is a pruned search, not a guarantee of the global optimum.</summary>
    /// <param name="logits">Frame-major logits including blank; NaN, positive infinity and all-masked rows are invalid.</param>
    /// <param name="vocabularySize">Columns, range 1..4096.</param>
    /// <param name="blankId">Blank column.</param>
    /// <param name="beamSize">Retained prefixes; default 8, range 1..64.</param>
    /// <param name="maxTokens">Prefix bound; default 1024, range 1..65536. An extension beyond this bound throws.</param>
    /// <param name="cancellationToken">Checked per frame and prefix expansion.</param>
    /// <exception cref="ArgumentException">Shape, bounds or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A prefix exceeds maxTokens.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static CtcBeamResult Decode(ReadOnlySpan<float> logits, int vocabularySize, int blankId, int beamSize = 8, int maxTokens = 1024, CancellationToken cancellationToken = default)
    {
        var result = ContextualCtcBeamSearch.Decode(logits, vocabularySize, blankId, EmptyContext, beamSize, maxTokens, cancellationToken);
        return new(result.Tokens, result.LogProbability);
    }
}
