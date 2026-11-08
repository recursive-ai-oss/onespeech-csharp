using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>Bounded greedy RNN-T and TDT search with token-level contextual shallow fusion.</summary>
/// <remarks>Context deltas are added to token log probabilities before selection, without renormalizing
/// the acoustic distribution. Blank does not advance context. TDT duration scores are unchanged.
/// Returned confidence remains the unmodified conditional acoustic probability, not a fused score.
/// Greedy decisions cannot recover a pruned alternative; in particular, partial-prefix credit can
/// affect an unfinished phrase at the end of the audio. Use conservative weights and measure controls.</remarks>
public static class ContextualTransducerSearch
{
    /// <summary>Greedy RNN-T with a phrase trie. Empty audio returns no tokens. Ties choose the smallest ID.</summary>
    /// <param name="frameCount">Nonnegative encoder-frame count.</param>
    /// <param name="vocabularySize">Columns including blank, 1..1000000.</param>
    /// <param name="blankId">Blank column, never present in phrases.</param>
    /// <param name="joint">Fills acoustic token logits; receives only accepted nonblank IDs.</param>
    /// <param name="phrases">Immutable context graph; empty is neutral.</param>
    /// <param name="maxSymbolsPerFrame">Nonblank bound per frame, default 10, range 1..1024.</param>
    /// <param name="maxTokens">Total emission bound, default 65536, range 1..1000000.</param>
    /// <param name="cancellationToken">Checked before each joint call.</param>
    /// <exception cref="ArgumentException">Dimensions, phrases, limits or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A symbol bound is exceeded.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public static ImmutableArray<FrameToken> DecodeRnnt(int frameCount, int vocabularySize, int blankId, TransducerLogits joint, PhraseTrie phrases, int maxSymbolsPerFrame = 10, int maxTokens = 65536, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        return TransducerGreedy.Decode(frameCount, vocabularySize, blankId, [], joint, maxSymbolsPerFrame, maxTokens, cancellationToken, phrases);
    }

    /// <summary>Greedy TDT with a phrase trie. Zero-duration blank advances one frame; intervals clip to the encoder end.</summary>
    /// <param name="frameCount">Nonnegative encoder-frame count.</param>
    /// <param name="vocabularySize">Columns including blank, 1..1000000.</param>
    /// <param name="blankId">Blank column, never present in phrases.</param>
    /// <param name="durations">Unique increasing frame skips starting at zero, 2..1024 entries, each at most 1024.</param>
    /// <param name="joint">Fills acoustic token and duration logits.</param>
    /// <param name="phrases">Immutable context graph; empty is neutral.</param>
    /// <param name="maxSymbolsPerFrame">Nonblank bound per frame, default 10, range 1..1024.</param>
    /// <param name="maxTokens">Total emission bound, default 65536, range 1..1000000.</param>
    /// <param name="cancellationToken">Checked before each joint call.</param>
    /// <exception cref="ArgumentException">Dimensions, phrases, limits, duration table or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A symbol bound is exceeded.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public static ImmutableArray<FrameToken> DecodeTdt(int frameCount, int vocabularySize, int blankId, ReadOnlySpan<int> durations, TransducerLogits joint, PhraseTrie phrases, int maxSymbolsPerFrame = 10, int maxTokens = 65536, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        TdtGreedy.ValidateDurations(durations);
        return TransducerGreedy.Decode(frameCount, vocabularySize, blankId, durations, joint, maxSymbolsPerFrame, maxTokens, cancellationToken, phrases);
    }
}
