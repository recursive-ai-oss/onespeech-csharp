namespace OneSpeech.Speakers;

/// <summary>Symmetric adaptive score normalization using the highest cosine cohort scores on each side.
/// Cohort vectors are copied into memory and are never persisted. Use disjoint, consented development speakers.</summary>
public sealed class AdaptiveScoreNormalizer
{
    private readonly float[][] cohort;
    private readonly int topCount;
    private readonly double deviationFloor;

    /// <summary>Copies a cohort of at least two equal-dimensional, finite, nonzero vectors.
    /// topCount defaults to 100 and is capped at cohort size; it must be at least two.
    /// deviationFloor defaults to 1e-6 and must be finite and positive. Invalid inputs throw ArgumentException.</summary>
    public AdaptiveScoreNormalizer(IReadOnlyList<ReadOnlyMemory<float>> cohort, int topCount = 100, double deviationFloor = 1e-6)
    {
        ArgumentNullException.ThrowIfNull(cohort);
        if (cohort.Count < 2 || topCount < 2) throw new ArgumentException("At least two cohort scores are required.", nameof(cohort));
        if (!double.IsFinite(deviationFloor) || deviationFloor <= 0) throw new ArgumentOutOfRangeException(nameof(deviationFloor));
        this.cohort = new float[cohort.Count][];
        int dimension = cohort[0].Length;
        for (int i = 0; i < cohort.Count; i++)
        {
            if (cohort[i].Length != dimension) throw new ArgumentException("Cohort dimensions differ.", nameof(cohort));
            this.cohort[i] = new float[dimension];
            EmbeddingMath.Normalize(cohort[i].Span, this.cohort[i]);
        }
        this.topCount = Math.Min(topCount, cohort.Count);
        this.deviationFloor = deviationFloor;
    }

    /// <summary>Embedding dimension fixed by the cohort at construction.</summary>
    public int Dimension => cohort[0].Length;

    /// <summary>Returns half the sum of the two cohort z-scores of the raw cosine similarity.
    /// Scores are unbounded, not probabilities. Uses population standard deviation, floored as configured.
    /// Invalid vectors throw ArgumentException; cancellation throws OperationCanceledException.</summary>
    public double Score(ReadOnlySpan<float> first, ReadOnlySpan<float> second, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        double raw = EmbeddingMath.Cosine(first, second);
        var a = Statistics(first, cancellationToken);
        var b = Statistics(second, cancellationToken);
        return 0.5 * ((raw - a.Mean) / a.Deviation + (raw - b.Mean) / b.Deviation);
    }

    private (double Mean, double Deviation) Statistics(ReadOnlySpan<float> vector, CancellationToken cancellationToken)
    {
        var scores = new double[cohort.Length];
        for (int i = 0; i < cohort.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            scores[i] = EmbeddingMath.Cosine(vector, cohort[i]);
        }
        Array.Sort(scores);
        double mean = 0, variance = 0;
        for (int i = scores.Length - topCount; i < scores.Length; i++) mean += scores[i] / topCount;
        for (int i = scores.Length - topCount; i < scores.Length; i++) variance += (scores[i] - mean) * (scores[i] - mean) / topCount;
        return (mean, Math.Max(deviationFloor, Math.Sqrt(variance)));
    }
}
