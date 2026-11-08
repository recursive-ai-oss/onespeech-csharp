using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>One owned autoregressive state for CPU beam search. Implementations may retain a model's KV cache.</summary>
public interface IBeamDecoderState : IDisposable
{
    /// <summary>Next-token logits in token-ID order. Length must remain fixed; negative infinity masks a token. NaNs and positive infinity are invalid.</summary>
    ReadOnlyMemory<float> Logits { get; }
    /// <summary>Returns a new independent owned state after accepting one non-EOS token. The parent remains valid until disposed. Cancellation must throw OperationCanceledException.</summary>
    IBeamDecoderState Advance(int token, CancellationToken cancellationToken);
}

/// <summary>A filter that mutates next-token logits for the generated prefix, excluding any model-conditioning prefix.</summary>
/// <param name="logits">Writable logits; use negative infinity to mask tokens.</param>
/// <param name="tokens">Previously generated token IDs, excluding EOS.</param>
public delegate void BeamLogitFilter(float[] logits, ReadOnlyMemory<int> tokens);

/// <summary>An immutable beam-search result. EOS is excluded from Tokens but its score is included when generated.</summary>
public sealed record BeamSearchResult
{
    /// <summary>Generated token IDs without EOS.</summary>
    public required ImmutableArray<int> Tokens { get; init; }
    /// <summary>Sum of filtered next-token natural log probabilities.</summary>
    public required double LogProbability { get; init; }
    /// <summary>Sum divided by the generated token count plus one, reserving one position for EOS.</summary>
    public double MeanLogProbability => LogProbability / (Tokens.Length + 1);
}

/// <summary>Bounded autoregressive beam search with deterministic score/token tie order.</summary>
public static class BeamSearch
{
    private sealed record Beam(int[] Tokens, double Sum, IBeamDecoderState State);
    private sealed record Candidate(int[] Tokens, double Sum, IBeamDecoderState Parent, int Token);

    /// <summary>Decodes up to maxTokens (default 224, range 1..1024) with beamSize active hypotheses (default 5, range 1..16).
    /// Completed hypotheses are ranked by mean natural log probability; the search stops after beamSize completions.</summary>
    /// <remarks>Takes ownership of initialState, including on failure. Each Advance must preserve the parent until it is disposed.
    /// The filter is applied before normalization. Vocabulary is limited to one million IDs. Working storage is bounded by beamSize and maxTokens.</remarks>
    /// <exception cref="ArgumentException">Limits, EOS, logits or the model's vocabulary shape are invalid.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested between expansions.</exception>
    public static BeamSearchResult Decode(IBeamDecoderState initialState, int endToken, int beamSize = 5, int maxTokens = 224, BeamLogitFilter? filter = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(initialState);
        List<Beam> active = [new([], 0, initialState)];
        List<BeamSearchResult> completed = [];
        try
        {
            int vocabulary = initialState.Logits.Length;
            if (beamSize is < 1 or > 16 || maxTokens is < 1 or > 1024 || vocabulary is < 1 or > 1000000 || (uint)endToken >= vocabulary) throw new ArgumentException("Invalid beam-search limits.");
            for (int step = 0; step < maxTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                List<Candidate> candidates = [];
                foreach (var beam in active)
                {
                    if (beam.State.Logits.Length != vocabulary) throw new ArgumentException("Beam vocabulary changed.");
                    float[] logits = beam.State.Logits.ToArray(); filter?.Invoke(logits, beam.Tokens);
                    double normalizer = LogitMath.LogSumExp(logits);
                    foreach (int token in LogitMath.TopIndices(logits, Math.Min(vocabulary, beamSize + 1)))
                    {
                        if (float.IsNegativeInfinity(logits[token])) continue;
                        double sum = beam.Sum + logits[token] - normalizer;
                        candidates.Add(new(token == endToken ? beam.Tokens : [.. beam.Tokens, token], sum, beam.State, token));
                    }
                }
                List<Candidate> selected = [];
                foreach (var candidate in candidates.OrderByDescending(c => c.Sum))
                {
                    if (candidate.Token == endToken) completed.Add(new BeamSearchResult { Tokens = candidate.Tokens.ToImmutableArray(), LogProbability = candidate.Sum });
                    else selected.Add(candidate);
                    if (selected.Count == beamSize) break;
                }
                if (completed.Count >= beamSize || selected.Count == 0) break;
                if (step == maxTokens - 1)
                {
                    completed.AddRange(selected.Select(c => new BeamSearchResult { Tokens = c.Tokens.ToImmutableArray(), LogProbability = c.Sum }));
                    break;
                }
                List<Beam> next = [];
                try
                {
                    foreach (var candidate in selected) next.Add(new(candidate.Tokens, candidate.Sum, candidate.Parent.Advance(candidate.Token, cancellationToken)));
                }
                catch { foreach (var beam in next) beam.State.Dispose(); throw; }
                foreach (var beam in active) beam.State.Dispose();
                active = next;
            }
            if (completed.Count == 0) throw new ArgumentException("No finite beam hypothesis remains.");
            return completed.OrderByDescending(c => c.MeanLogProbability).First();
        }
        finally { foreach (var beam in active) beam.State.Dispose(); }
    }
}

/// <summary>Stable next-token selection and natural-log normalization used by greedy and beam decoding.</summary>
public static class LogitMath
{
    /// <summary>Returns up to count IDs ordered by descending logit, breaking exact ties by smaller ID. Count must be 1..17 and no larger than the vocabulary.</summary>
    /// <exception cref="ArgumentException">Shape or logits are invalid.</exception>
    public static int[] TopIndices(ReadOnlySpan<float> logits, int count)
    {
        if (count < 1 || count > 17 || count > logits.Length) throw new ArgumentException("Invalid top-token count.");
        int[] top = new int[count]; Array.Fill(top, -1);
        for (int token = 0; token < logits.Length; token++)
        {
            float value = logits[token]; if (float.IsNaN(value) || float.IsPositiveInfinity(value)) throw new ArgumentException("Logits contain NaN or positive infinity.", nameof(logits));
            for (int position = 0; position < count; position++)
                if (top[position] < 0 || value > logits[top[position]])
                {
                    for (int j = count - 1; j > position; j--) top[j] = top[j - 1];
                    top[position] = token; break;
                }
        }
        return top;
    }
    /// <summary>Returns the greatest-logit ID, with smaller IDs winning ties. Throws ArgumentException when no finite candidate exists.</summary>
    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        int[] top = TopIndices(logits, 1);
        if (!float.IsFinite(logits[top[0]])) throw new ArgumentException("No finite token remains.", nameof(logits));
        return top[0];
    }
    /// <summary>Returns log(sum(exp(logits))) using a stable maximum shift; negative infinity is a permitted mask. An all-masked or empty vector returns negative infinity.</summary>
    /// <exception cref="ArgumentException">A logit is NaN or positive infinity.</exception>
    public static double LogSumExp(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits) { if (float.IsNaN(value) || float.IsPositiveInfinity(value)) throw new ArgumentException("Invalid logit.", nameof(logits)); max = Math.Max(max, value); }
        if (float.IsNegativeInfinity(max)) return double.NegativeInfinity;
        double sum = 0; foreach (float value in logits) sum += Math.Exp(value - max);
        return max + Math.Log(sum);
    }
}
