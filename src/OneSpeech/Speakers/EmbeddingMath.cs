namespace OneSpeech.Speakers;

/// <summary>Finite-vector arithmetic for speaker scores. These functions do not retain their inputs.</summary>
public static class EmbeddingMath
{
    /// <summary>Returns cosine similarity in [-1,1]. Throws ArgumentException for empty, unequal,
    /// nonfinite or zero-norm vectors. Accumulates in double precision.</summary>
    public static double Cosine(ReadOnlySpan<float> first, ReadOnlySpan<float> second)
    {
        if (first.Length != second.Length) throw new ArgumentException("Embedding dimensions differ.");
        double a = SquaredNorm(first), b = SquaredNorm(second), dot = 0;
        for (int i = 0; i < first.Length; i++) dot += (double)first[i] * second[i];
        return Math.Clamp(dot / Math.Sqrt(a * b), -1, 1);
    }

    /// <summary>Copies a unit-length vector into destination, which must have the same length.
    /// In-place use is supported. Throws ArgumentException for empty, nonfinite or zero-norm input.</summary>
    public static void Normalize(ReadOnlySpan<float> embedding, Span<float> destination)
    {
        if (embedding.Length != destination.Length) throw new ArgumentException("Embedding dimensions differ.");
        double norm = Math.Sqrt(SquaredNorm(embedding));
        for (int i = 0; i < embedding.Length; i++) destination[i] = (float)(embedding[i] / norm);
    }

    /// <summary>Averages equally weighted, individually unit-normalized vectors and normalizes the result.
    /// Throws ArgumentException for an empty collection, invalid vectors, unequal dimensions or a cancelling mean.
    /// Returns new caller-owned storage; does not retain inputs.</summary>
    public static float[] Average(IReadOnlyList<ReadOnlyMemory<float>> embeddings)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        if (embeddings.Count == 0) throw new ArgumentException("At least one embedding is required.", nameof(embeddings));
        int dimension = embeddings[0].Length;
        var sum = new double[dimension];
        foreach (var embedding in embeddings)
        {
            if (embedding.Length != dimension) throw new ArgumentException("Embedding dimensions differ.", nameof(embeddings));
            double norm = Math.Sqrt(SquaredNorm(embedding.Span));
            for (int i = 0; i < dimension; i++) sum[i] += embedding.Span[i] / norm;
        }
        var result = new float[dimension];
        for (int i = 0; i < dimension; i++) result[i] = (float)(sum[i] / embeddings.Count);
        Normalize(result, result);
        return result;
    }

    private static double SquaredNorm(ReadOnlySpan<float> embedding)
    {
        if (embedding.IsEmpty) throw new ArgumentException("An embedding must not be empty.", nameof(embedding));
        double sum = 0;
        foreach (float value in embedding)
        {
            if (!float.IsFinite(value)) throw new ArgumentException("Embeddings must be finite.", nameof(embedding));
            sum += (double)value * value;
        }
        if (sum == 0) throw new ArgumentException("An embedding must have nonzero norm.", nameof(embedding));
        return sum;
    }
}
