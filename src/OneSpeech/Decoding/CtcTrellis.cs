using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A target token's occupied frame interval in a successful CTC maximum-probability path.</summary>
/// <param name="TokenIndex">Zero-based position in the target, distinguishing repeated tokens.</param>
/// <param name="TokenId">Vocabulary ID, never the blank ID.</param>
/// <param name="StartFrame">Inclusive frame index, relative to the supplied emission matrix.</param>
/// <param name="EndFrame">Exclusive frame index. Blank frames are not included.</param>
public readonly record struct CtcTokenSpan(int TokenIndex, int TokenId, int StartFrame, int EndFrame);

/// <summary>The maximum-path CTC alignment, not the sum over all paths or a posterior confidence.</summary>
public sealed class CtcAlignment
{
    internal CtcAlignment(bool succeeded, double logScore, ImmutableArray<CtcTokenSpan> tokens)
    {
        Succeeded = succeeded;
        LogScore = logScore;
        Tokens = tokens;
    }

    /// <summary>False when no finite-probability path consumes the complete target. Failed alignment has no token intervals.</summary>
    public bool Succeeded { get; }

    /// <summary>Sum of natural-log frame probabilities along the best path, or negative infinity on failure. Empty audio with an empty target has score zero.</summary>
    public double LogScore { get; }

    /// <summary>One interval per target token, in target order; empty on failure or for an empty target. No audio-time conversion or blank-duration attribution is performed.</summary>
    public ImmutableArray<CtcTokenSpan> Tokens { get; }
}

/// <summary>Viterbi forced alignment on a blank-expanded CTC target, with bounded traceback memory.</summary>
public static class CtcTrellis
{
    /// <summary>Finds the most probable complete CTC path for a fixed target in row-major natural-log probabilities.</summary>
    /// <param name="logProbabilities">Exactly frameCount × vocabularySize values, each nonpositive and either finite or negative infinity. Row normalization is the caller's responsibility.</param>
    /// <param name="frameCount">Number of frames, zero or greater.</param>
    /// <param name="vocabularySize">Positive vocabulary size including blank.</param>
    /// <param name="target">Nonblank IDs in the required order. Adjacent equal IDs require at least one intervening blank frame.</param>
    /// <param name="blankId">Blank ID in [0, vocabularySize).</param>
    /// <param name="cancellationToken">Checked before validation and at least once per frame, including traceback.</param>
    /// <returns>A complete alignment or an explicit failed result, never a partial target. Empty targets are scored as the all-blank path.</returns>
    /// <remarks>Every frame is consumed. Exact predecessor ties prefer staying, then advancing one state, then advancing two; terminal ties prefer the last token over trailing blank. The frameCount × (2 × target.Length + 1) traceback has a limit of 16 million cells; score storage uses two state rows. These are token occupancy boundaries, not calibrated acoustic word boundaries.</remarks>
    /// <exception cref="ArgumentException">The emission shape, target IDs, log probabilities, or trellis size are invalid.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public static CtcAlignment Align(ReadOnlySpan<float> logProbabilities, int frameCount, int vocabularySize,
        ReadOnlySpan<int> target, int blankId = 0, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (frameCount < 0 || vocabularySize <= 0 || (long)frameCount * vocabularySize != logProbabilities.Length)
            throw new ArgumentException("Invalid CTC emission shape.", nameof(logProbabilities));
        if (blankId < 0 || blankId >= vocabularySize)
            throw new ArgumentOutOfRangeException(nameof(blankId));
        long stateCount = 2L * target.Length + 1;
        if (stateCount > 16000000 || stateCount * frameCount > 16000000)
            throw new ArgumentException("CTC traceback exceeds 16 million cells.", nameof(target));
        int minimumFrames = target.Length;
        for (int i = 0; i < target.Length; i++)
        {
            if (target[i] < 0 || target[i] >= vocabularySize || target[i] == blankId)
                throw new ArgumentException("CTC targets must contain valid nonblank IDs.", nameof(target));
            if (i > 0 && target[i] == target[i - 1]) minimumFrames++;
        }
        for (int frame = 0; frame < frameCount; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (float value in logProbabilities.Slice(frame * vocabularySize, vocabularySize))
                if (float.IsNaN(value) || value > 0)
                    throw new ArgumentException("CTC inputs must be nonpositive natural-log probabilities.", nameof(logProbabilities));
        }
        if (frameCount < minimumFrames) return new CtcAlignment(false, double.NegativeInfinity, []);
        if (target.IsEmpty)
        {
            double score = 0;
            for (int frame = 0; frame < frameCount; frame++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                score += logProbabilities[frame * vocabularySize + blankId];
            }
            return new CtcAlignment(double.IsFinite(score), score, []);
        }

        // Graves et al. 2006 §4.1: insert blank at both ends and between tokens.
        // Replace sum with maximum in Eq. 6–7 to recover one forced-alignment path.
        int states = (int)stateCount;
        double[] previous = new double[states], current = new double[states];
        byte[] trace = new byte[frameCount * states];
        Array.Fill(previous, double.NegativeInfinity);
        previous[0] = logProbabilities[blankId];
        previous[1] = logProbabilities[target[0]];
        for (int frame = 1; frame < frameCount; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int state = 0; state < states; state++)
            {
                bool isBlank = (state & 1) == 0;
                int token = isBlank ? blankId : target[state / 2];
                double best = previous[state];
                byte step = 0;
                if (state > 0 && previous[state - 1] > best) { best = previous[state - 1]; step = 1; }
                if (!isBlank && state > 1 && token != target[state / 2 - 1] && previous[state - 2] > best)
                {
                    best = previous[state - 2]; step = 2;
                }
                current[state] = best + logProbabilities[frame * vocabularySize + token];
                trace[frame * states + state] = step;
            }
            (previous, current) = (current, previous);
        }
        int terminal = states - 2;
        if (previous[states - 1] > previous[terminal]) terminal = states - 1;
        double logScore = previous[terminal];
        if (!double.IsFinite(logScore)) return new CtcAlignment(false, double.NegativeInfinity, []);

        int[] starts = new int[target.Length], ends = new int[target.Length];
        for (int frame = frameCount - 1; frame >= 0; frame--)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if ((terminal & 1) != 0)
            {
                int tokenIndex = terminal / 2;
                starts[tokenIndex] = frame;
                if (ends[tokenIndex] == 0) ends[tokenIndex] = frame + 1;
            }
            terminal -= trace[frame * states + terminal];
        }
        var tokens = ImmutableArray.CreateBuilder<CtcTokenSpan>(target.Length);
        for (int i = 0; i < target.Length; i++) tokens.Add(new CtcTokenSpan(i, target[i], starts[i], ends[i]));
        return new CtcAlignment(true, logScore, tokens.MoveToImmutable());
    }
}
