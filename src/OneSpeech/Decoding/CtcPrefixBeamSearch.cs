using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A CTC collapsed sequence and its summed natural log probability over retained paths.</summary>
/// <param name="Tokens">Nonblank token IDs.</param>
/// <param name="LogProbability">Natural log probability; pruning can discard contributing paths.</param>
public sealed record CtcBeamResult(ImmutableArray<int> Tokens, double LogProbability);

/// <summary>CTC prefix beam search, summing blank/nonblank path probabilities separately before pruning.</summary>
public static class CtcPrefixBeamSearch
{
    private sealed record Prefix(int[] Tokens, double Blank, double Nonblank)
    {
        internal double Score => Add(Blank, Nonblank);
    }

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
        GreedyMath.ValidateVocabulary(vocabularySize, blankId);
        if (vocabularySize > 4096 || logits.Length % vocabularySize != 0 || beamSize is < 1 or > 64 || maxTokens is < 1 or > 65536) throw new ArgumentException("Invalid prefix beam dimensions or limits.");
        cancellationToken.ThrowIfCancellationRequested();
        List<Prefix> active = [new([], 0, double.NegativeInfinity)];
        for (int t = 0; t < logits.Length / vocabularySize; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var row = logits.Slice(t * vocabularySize, vocabularySize);
            int best = GreedyMath.ArgMax(row);
            double sum = 0;
            foreach (float value in row) sum += Math.Exp((double)value - row[best]);
            double normalizer = Math.Log(sum) + row[best];
            Dictionary<string, Prefix> next = new(StringComparer.Ordinal);
            void Accumulate(int[] tokens, double blank, double nonblank)
            {
                string key = string.Join(',', tokens);
                if (next.TryGetValue(key, out var previous)) next[key] = new(previous.Tokens, Add(previous.Blank, blank), Add(previous.Nonblank, nonblank));
                else next.Add(key, new(tokens, blank, nonblank));
            }
            foreach (var prefix in active)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Accumulate(prefix.Tokens, prefix.Score + row[blankId] - normalizer, double.NegativeInfinity);
                for (int token = 0; token < vocabularySize; token++)
                {
                    if (token == blankId || float.IsNegativeInfinity(row[token])) continue;
                    double logp = row[token] - normalizer;
                    bool repeat = prefix.Tokens.Length > 0 && prefix.Tokens[^1] == token;
                    if (repeat) Accumulate(prefix.Tokens, double.NegativeInfinity, prefix.Nonblank + logp);
                    double extension = (repeat ? prefix.Blank : prefix.Score) + logp;
                    if (double.IsNegativeInfinity(extension)) continue;
                    if (prefix.Tokens.Length == maxTokens) throw new InvalidOperationException("CTC prefix bound exceeded.");
                    Accumulate([.. prefix.Tokens, token], double.NegativeInfinity, extension);
                }
            }
            active = next.Values.Where(p => !double.IsNegativeInfinity(p.Score)).ToList();
            active.Sort(Compare);
            if (active.Count > beamSize) active.RemoveRange(beamSize, active.Count - beamSize);
        }
        var winner = active[0];
        return new(winner.Tokens.ToImmutableArray(), winner.Score);
    }

    private static int Compare(Prefix left, Prefix right)
    {
        int score = right.Score.CompareTo(left.Score);
        if (score != 0) return score;
        for (int i = 0; i < Math.Min(left.Tokens.Length, right.Tokens.Length); i++)
        {
            int token = left.Tokens[i].CompareTo(right.Tokens[i]);
            if (token != 0) return token;
        }
        return left.Tokens.Length.CompareTo(right.Tokens.Length);
    }
    private static double Add(double a, double b)
    {
        double largest = Math.Max(a, b);
        return double.IsNegativeInfinity(largest) ? largest : largest + Math.Log(1 + Math.Exp(Math.Min(a, b) - largest));
    }
}
