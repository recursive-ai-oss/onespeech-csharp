using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A contextual CTC result with separate acoustic and completed-phrase scores.</summary>
/// <param name="Tokens">Collapsed nonblank IDs.</param>
/// <param name="LogProbability">Summed natural log acoustic probability of retained paths; pruning loses paths.</param>
/// <param name="ContextScore">Natural-log reward for completed phrases only.</param>
public sealed record ContextualCtcResult(ImmutableArray<int> Tokens, double LogProbability, double ContextScore);

/// <summary>CTC prefix beam search with inference-only phrase shallow fusion before pruning.</summary>
public static class ContextualCtcBeamSearch
{
    private sealed record Prefix(int[] Tokens, double Blank, double Nonblank, int State, double Context)
    {
        internal double Acoustic => Add(Blank, Nonblank);
        internal double Score => Acoustic + Context;
    }

    /// <summary>Decodes frame-major logits; blank and repeated frames never advance the phrase trie.
    /// Partial-prefix credit is repaid before final ranking. Ties select lexicographically smaller IDs.
    /// Empty audio returns the empty sequence with scores zero. Search is pruned, not globally exact.</summary>
    /// <param name="logits">Frame-major logits; NaN, positive infinity and all-masked rows are refused.</param>
    /// <param name="vocabularySize">Columns including blank, 1..4096.</param>
    /// <param name="blankId">Blank column; must not appear in any contextual phrase.</param>
    /// <param name="phrases">Context graph; an empty graph is neutral.</param>
    /// <param name="beamSize">Retained prefixes, default 8, range 1..64.</param>
    /// <param name="maxTokens">Nonblank prefix bound, default 1024, range 1..65536.</param>
    /// <param name="cancellationToken">Checked per frame and prefix.</param>
    /// <exception cref="ArgumentException">Shape, bounds or logits are invalid.</exception>
    /// <exception cref="InvalidOperationException">A prefix exceeds the bound.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public static ContextualCtcResult Decode(ReadOnlySpan<float> logits, int vocabularySize, int blankId, PhraseTrie phrases, int beamSize = 8, int maxTokens = 1024, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        GreedyMath.ValidateVocabulary(vocabularySize, blankId);
        phrases.ValidateVocabulary(vocabularySize, blankId);
        if (vocabularySize > 4096 || logits.Length % vocabularySize != 0 || beamSize is < 1 or > 64 || maxTokens is < 1 or > 65536) throw new ArgumentException("Invalid contextual beam dimensions or limits.");
        cancellationToken.ThrowIfCancellationRequested();
        List<Prefix> active = [new([], 0, double.NegativeInfinity, 0, 0)];
        for (int t = 0; t < logits.Length / vocabularySize; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var row = logits.Slice(t * vocabularySize, vocabularySize);
            int best = GreedyMath.ArgMax(row);
            double sum = 0;
            foreach (float value in row) sum += Math.Exp((double)value - row[best]);
            double normalizer = Math.Log(sum) + row[best];
            Dictionary<string, Prefix> next = new(StringComparer.Ordinal);
            void Accumulate(int[] tokens, double blank, double nonblank, int state, double context)
            {
                string key = string.Join(',', tokens);
                if (next.TryGetValue(key, out var previous)) next[key] = previous with { Blank = Add(previous.Blank, blank), Nonblank = Add(previous.Nonblank, nonblank) };
                else next.Add(key, new(tokens, blank, nonblank, state, context));
            }
            foreach (var prefix in active)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Accumulate(prefix.Tokens, prefix.Acoustic + row[blankId] - normalizer, double.NegativeInfinity, prefix.State, prefix.Context);
                for (int token = 0; token < vocabularySize; token++)
                {
                    if (token == blankId || float.IsNegativeInfinity(row[token])) continue;
                    double logp = row[token] - normalizer;
                    bool repeat = prefix.Tokens.Length > 0 && prefix.Tokens[^1] == token;
                    if (repeat) Accumulate(prefix.Tokens, double.NegativeInfinity, prefix.Nonblank + logp, prefix.State, prefix.Context);
                    double extension = (repeat ? prefix.Blank : prefix.Acoustic) + logp;
                    if (double.IsNegativeInfinity(extension)) continue;
                    if (prefix.Tokens.Length == maxTokens) throw new InvalidOperationException("CTC prefix bound exceeded.");
                    double delta = phrases.Advance(prefix.State, token, out int state);
                    Accumulate([.. prefix.Tokens, token], double.NegativeInfinity, extension, state, prefix.Context + delta);
                }
            }
            active = next.Values.Where(p => !double.IsNegativeInfinity(p.Acoustic)).ToList();
            if (t + 1 == logits.Length / vocabularySize)
                active = active.Select(p => p with { Context = p.Context + phrases.Finish(p.State) }).ToList();
            active.Sort(Compare);
            if (active.Count > beamSize) active.RemoveRange(beamSize, active.Count - beamSize);
        }
        var winner = active[0];
        return new(winner.Tokens.ToImmutableArray(), winner.Acoustic, winner.Context);
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
