namespace OneSpeech.Speakers;

/// <summary>Bounded, causal cosine-centroid clustering. IDs are allocated in first-observation order and never merged or reused. Not thread safe; retains only centroids, never input vectors.</summary>
public sealed class OnlineSpeakerClusterer
{
    private readonly float[] centroids;
    private readonly float[] normalized;
    private readonly int[] counts;
    private readonly SpeakerLabel[] labels;
    private readonly double threshold;
    private int count;

    /// <summary>Creates a clusterer. Dimension must be 1–4096, capacity 1–128 (default 16), minimum cosine similarity in [-1,1] (default 0.3). Prefix defaults to speaker. Throws ArgumentOutOfRangeException for invalid bounds or ArgumentException for an empty prefix. At capacity an unmatched vector is Unassigned, not forcibly merged.</summary>
    public OnlineSpeakerClusterer(int dimension, int capacity = 16, double minimumSimilarity = 0.3, string prefix = "speaker")
    {
        if (dimension is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(dimension));
        if (capacity is < 1 or > 128) throw new ArgumentOutOfRangeException(nameof(capacity));
        if (!double.IsFinite(minimumSimilarity) || minimumSimilarity is < -1 or > 1) throw new ArgumentOutOfRangeException(nameof(minimumSimilarity));
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        threshold = minimumSimilarity;
        normalized = new float[dimension];
        centroids = new float[dimension * capacity];
        counts = new int[capacity];
        labels = new SpeakerLabel[capacity];
        for (int i = 0; i < capacity; i++) labels[i] = new SpeakerLabel { Id = prefix + "-" + (i + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) };
    }

    /// <summary>Current number of allocated IDs, bounded by constructor capacity.</summary>
    public int Count => count;

    /// <summary>Assigns and updates from one finite, nonzero vector of the configured dimension. No allocations. Ties prefer the oldest ID. Each observation has equal weight until 100 observations, then an exponential update of 1/101. Invalid input throws ArgumentException without changing state. No future observations revise this assignment.</summary>
    public SpeakerLabel Assign(ReadOnlySpan<float> embedding)
    {
        EmbeddingMath.Normalize(embedding, normalized);
        int best = -1;
        double similarity = threshold;
        for (int i = 0; i < count; i++)
        {
            double score = EmbeddingMath.Cosine(normalized, centroids.AsSpan(i * normalized.Length, normalized.Length));
            if (score >= similarity && (best < 0 || score > similarity)) { best = i; similarity = score; }
        }
        if (best < 0)
        {
            if (count == labels.Length) return SpeakerLabel.Unassigned;
            best = count++;
            normalized.CopyTo(centroids.AsSpan(best * normalized.Length, normalized.Length));
            counts[best] = 1;
        }
        else
        {
            var centroid = centroids.AsSpan(best * normalized.Length, normalized.Length);
            int weight = counts[best];
            // Avoid cancellation for opposite vectors when a caller permits negative similarity.
            if (similarity > -0.999999)
            {
                for (int i = 0; i < centroid.Length; i++) centroid[i] = (centroid[i] * weight + normalized[i]) / (weight + 1);
                EmbeddingMath.Normalize(centroid, centroid);
                counts[best] = Math.Min(100, weight + 1);
            }
        }
        return labels[best];
    }

    /// <summary>Erases retained biometric centroids. IDs restart at one; call only after the session ends. No embeddings are persisted.</summary>
    public void Clear() { Array.Clear(centroids); Array.Clear(normalized); Array.Clear(counts); count = 0; }
}
