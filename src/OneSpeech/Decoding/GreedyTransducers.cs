using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A decoded nonblank token with an encoder-frame interval and token probability.</summary>
/// <param name="Id">Vocabulary token ID.</param>
/// <param name="StartFrame">Inclusive encoder frame, starting at zero.</param>
/// <param name="EndFrame">Exclusive encoder frame. TDT zero-duration emissions can have equal boundaries.</param>
/// <param name="Confidence">Conditional token probability in [0,1], excluding the duration probability.</param>
public readonly record struct FrameToken(int Id, int StartFrame, int EndFrame, float Confidence);

/// <summary>Fills joint logits for one encoder frame and the accepted nonblank prefix. No softmax is required.</summary>
/// <remarks>The prefix and output spans are valid only during the call. Implementations must fill every output element,
/// preserve the prefix, and honor cancellation. Duration logits are empty for RNN-T. Predictor state may be cached by prefix length.</remarks>
/// <param name="frame">Zero-based encoder frame.</param>
/// <param name="prefix">Accepted token IDs, excluding blanks.</param>
/// <param name="tokenLogits">Token logits including blank.</param>
/// <param name="durationLogits">Separate TDT duration-head logits in the caller's duration-table order.</param>
/// <param name="cancellationToken">Cancels model execution.</param>
public delegate void TransducerLogits(int frame, ReadOnlySpan<int> prefix, Span<float> tokenLogits, Span<float> durationLogits, CancellationToken cancellationToken);

/// <summary>Deterministic greedy CTC collapse. Ties select the smallest token ID.</summary>
public static class CtcGreedy
{
    /// <summary>Collapses frame-major logits, merging consecutive identical IDs before removing blanks.
    /// Empty input returns no tokens. Each run's confidence is its mean conditional probability.</summary>
    /// <param name="logits">Frame-major matrix, including blank.</param>
    /// <param name="vocabularySize">Columns, in [1,1000000].</param>
    /// <param name="blankId">Blank column.</param>
    /// <param name="cancellationToken">Checked per frame.</param>
    /// <exception cref="ArgumentException">Dimensions, blank ID, or logits are invalid. NaN, positive infinity and all-masked rows are rejected.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static ImmutableArray<FrameToken> Decode(ReadOnlySpan<float> logits, int vocabularySize, int blankId, CancellationToken cancellationToken = default)
    {
        GreedyMath.ValidateVocabulary(vocabularySize, blankId);
        if (logits.Length % vocabularySize != 0) throw new ArgumentException("Logits must contain whole frames.", nameof(logits));
        cancellationToken.ThrowIfCancellationRequested();
        var result = ImmutableArray.CreateBuilder<FrameToken>();
        int previous = blankId, start = 0, count = 0;
        double probability = 0;
        for (int t = 0; t < logits.Length / vocabularySize; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var row = logits.Slice(t * vocabularySize, vocabularySize);
            int id = GreedyMath.ArgMax(row);
            if (id != previous && previous != blankId)
            {
                result.Add(new(previous, start, t, (float)(probability / count)));
                probability = 0; count = 0;
            }
            if (id != blankId)
            {
                if (id != previous) start = t;
                probability += GreedyMath.Probability(row, id); count++;
            }
            previous = id;
        }
        if (previous != blankId) result.Add(new(previous, start, logits.Length / vocabularySize, (float)(probability / count)));
        return result.ToImmutable();
    }
}

/// <summary>Greedy RNN-T search from Xu et al. 2023, Algorithm 1. Nonblanks retain the current frame; blank advances it.</summary>
public static class RnntGreedy
{
    /// <summary>Decodes with explicit resource bounds. Uses one-frame token intervals (frame resolution, not measured phonetic durations).</summary>
    /// <param name="frameCount">Encoder frames, zero or greater.</param>
    /// <param name="vocabularySize">Token columns including blank, in [1,1000000].</param>
    /// <param name="blankId">Blank token ID.</param>
    /// <param name="joint">Fills token logits for the current frame and accepted prefix.</param>
    /// <param name="maxSymbolsPerFrame">Nonblank limit per frame, default 10; range 1..1024. Exceeding it throws rather than dropping tokens.</param>
    /// <param name="maxTokens">Total nonblank limit, default 65536; range 1..1000000.</param>
    /// <param name="cancellationToken">Checked before each joint call.</param>
    /// <exception cref="ArgumentException">Dimensions, limits or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A symbol or token bound was exceeded.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static ImmutableArray<FrameToken> Decode(int frameCount, int vocabularySize, int blankId, TransducerLogits joint, int maxSymbolsPerFrame = 10, int maxTokens = 65536, CancellationToken cancellationToken = default)
        => TransducerGreedy.Decode(frameCount, vocabularySize, blankId, [], joint, maxSymbolsPerFrame, maxTokens, cancellationToken);
}

/// <summary>Greedy token-and-duration transducer search from Xu et al. 2023, Algorithm 2.</summary>
public static class TdtGreedy
{
    /// <summary>Decodes separate token/duration heads. Nonblank emissions update the prefix; every prediction advances by its duration.
    /// A blank with duration zero advances one frame to prevent a stalled search. Duration intervals are clipped at the encoder end.</summary>
    /// <param name="frameCount">Encoder frames, zero or greater.</param>
    /// <param name="vocabularySize">Token columns including blank, in [1,1000000].</param>
    /// <param name="blankId">Blank ID.</param>
    /// <param name="durations">Unique increasing frame skips, starting with zero; 2..1024 entries, each at most 1024.</param>
    /// <param name="joint">Fills token and duration logits for the current frame and prefix.</param>
    /// <param name="maxSymbolsPerFrame">Nonblank bound at a frame; default 10, range 1..1024.</param>
    /// <param name="maxTokens">Total nonblank bound; default 65536, range 1..1000000.</param>
    /// <param name="cancellationToken">Checked before each joint call.</param>
    /// <exception cref="ArgumentException">Dimensions, duration table, limits or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A symbol or token bound was exceeded.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static ImmutableArray<FrameToken> Decode(int frameCount, int vocabularySize, int blankId, ReadOnlySpan<int> durations, TransducerLogits joint, int maxSymbolsPerFrame = 10, int maxTokens = 65536, CancellationToken cancellationToken = default)
    {
        ValidateDurations(durations);
        return TransducerGreedy.Decode(frameCount, vocabularySize, blankId, durations, joint, maxSymbolsPerFrame, maxTokens, cancellationToken);
    }

    internal static void ValidateDurations(ReadOnlySpan<int> durations)
    {
        if (durations.Length is < 2 or > 1024 || durations[0] != 0) throw new ArgumentException("Durations must start at zero and contain 2..1024 entries.", nameof(durations));
        for (int i = 1; i < durations.Length; i++)
            if (durations[i] <= durations[i - 1] || durations[i] > 1024) throw new ArgumentException("Durations must be increasing and at most 1024 frames.", nameof(durations));
    }
}

internal static class TransducerGreedy
{
    internal static ImmutableArray<FrameToken> Decode(int frames, int vocabulary, int blank, ReadOnlySpan<int> durations, TransducerLogits joint, int perFrame, int maxTokens, CancellationToken ct, PhraseTrie? phrases = null)
    {
        ArgumentNullException.ThrowIfNull(joint);
        GreedyMath.ValidateVocabulary(vocabulary, blank);
        phrases?.ValidateVocabulary(vocabulary, blank);
        if (frames < 0 || perFrame is < 1 or > 1024 || maxTokens is < 1 or > 1000000) throw new ArgumentException("Invalid transducer bounds.");
        ct.ThrowIfCancellationRequested();
        if (frames == 0) return [];
        float[] logits = new float[vocabulary], durationLogits = new float[durations.Length];
        int[] prefix = new int[maxTokens];
        var result = ImmutableArray.CreateBuilder<FrameToken>();
        int frame = 0, symbols = 0, phraseState = 0;
        while (frame < frames)
        {
            ct.ThrowIfCancellationRequested();
            // Poison outputs so a callback that forgets to fill a column cannot reuse stale logits.
            Array.Fill(logits, float.NaN); Array.Fill(durationLogits, float.NaN);
            joint(frame, prefix.AsSpan(0, result.Count), logits, durationLogits, ct);
            int id = GreedyMath.ArgMax(logits);
            if (phrases != null)
            {
                double best = double.NegativeInfinity;
                for (int token = 0; token < vocabulary; token++)
                {
                    double score = logits[token] + (token == blank ? 0 : phrases.Advance(phraseState, token, out _));
                    if (score > best) { best = score; id = token; }
                }
            }
            int skip = durations.IsEmpty ? (id == blank ? 1 : 0) : durations[GreedyMath.ArgMax(durationLogits)];
            if (id != blank)
            {
                if (symbols == perFrame || result.Count == maxTokens) throw new InvalidOperationException("Transducer symbol bound exceeded.");
                if (phrases != null) { phrases.Advance(phraseState, id, out int next); phraseState = next; }
                prefix[result.Count] = id;
                result.Add(new(id, frame, (int)Math.Min(frames, (long)frame + (durations.IsEmpty ? 1 : skip)), GreedyMath.Probability(logits, id)));
                symbols++;
            }
            else skip = Math.Max(1, skip);
            if (skip > 0) { frame = (int)Math.Min(frames, (long)frame + skip); symbols = 0; }
        }
        return result.ToImmutable();
    }
}

internal static class GreedyMath
{
    internal static void ValidateVocabulary(int count, int blank)
    {
        if (count is < 1 or > 1000000 || (uint)blank >= count) throw new ArgumentException("Invalid vocabulary or blank ID.");
    }
    internal static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (float.IsNaN(values[i]) || float.IsPositiveInfinity(values[i])) throw new ArgumentException("Invalid logits.");
            if (values[i] > values[best]) best = i;
        }
        if (values.IsEmpty || float.IsNegativeInfinity(values[best])) throw new ArgumentException("All logits are masked.");
        return best;
    }
    internal static float Probability(ReadOnlySpan<float> values, int selected)
    {
        double sum = 0;
        foreach (float value in values) sum += Math.Exp((double)value - values[selected]);
        return (float)(1 / sum);
    }
}
