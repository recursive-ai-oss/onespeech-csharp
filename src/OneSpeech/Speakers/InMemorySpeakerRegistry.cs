namespace OneSpeech.Speakers;

/// <summary>Thread-safe in-memory enrollments with cosine scoring, optionally cohort-normalized.
/// Enrollment replaces a name; no embedding, cohort or audio is serialized. Dispose clears enrollments.
/// The caller owns the embedder and must keep it alive until this registry is disposed.</summary>
public sealed class InMemorySpeakerRegistry : SpeakerRegistry, IDisposable
{
    private readonly SpeakerEmbedder embedder;
    private readonly AdaptiveScoreNormalizer? normalizer;
    private readonly Dictionary<string, float[]> enrollments = new(StringComparer.Ordinal);
    private readonly object sync = new();
    private bool disposed;

    /// <summary>Creates an empty registry using a borrowed embedder, an explicit acceptance threshold,
    /// and optional AS-norm cohort. Threshold is finite, and in [-1,1] for raw cosine.
    /// No universal default threshold is supplied: select one on disjoint development trials.
    /// Invalid dimensions or threshold throw ArgumentException.</summary>
    public InMemorySpeakerRegistry(SpeakerEmbedder embedder, double threshold, AdaptiveScoreNormalizer? normalizer = null)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        if (embedder.Dimension <= 0) throw new ArgumentException("Embedder dimension must be positive.", nameof(embedder));
        if (normalizer is not null && normalizer.Dimension != embedder.Dimension)
            throw new ArgumentException("Cohort and embedder dimensions differ.", nameof(normalizer));
        if (!double.IsFinite(threshold) || (normalizer is null && (threshold < -1 || threshold > 1))) throw new ArgumentOutOfRangeException(nameof(threshold));
        this.embedder = embedder;
        this.normalizer = normalizer;
        Threshold = threshold;
    }

    /// <summary>Inclusive identification acceptance threshold on this registry's score scale.</summary>
    public double Threshold { get; }

    /// <summary>Enrolls or replaces a case-sensitive, nonblank ID from one utterance.
    /// Throws ArgumentException for invalid input, OperationCanceledException on cancellation,
    /// or ObjectDisposedException after disposal. Existing enrollment survives failed extraction.</summary>
    public override void Enroll(string speakerId, AudioBuffer audio, CancellationToken cancellationToken)
        => Enroll(speakerId, new[] { audio }, cancellationToken);

    /// <summary>Replaces an ID with the normalized equal-weight average of individually normalized utterances.
    /// At least one utterance is required. Failure or cancellation leaves the old enrollment unchanged.
    /// Throws ArgumentException for invalid IDs, audio or embeddings, or ObjectDisposedException after disposal.</summary>
    public void Enroll(string speakerId, IReadOnlyList<AudioBuffer> utterances, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(speakerId);
        ArgumentNullException.ThrowIfNull(utterances);
        if (utterances.Count == 0) throw new ArgumentException("At least one utterance is required.", nameof(utterances));
        lock (sync)
        {
            Check(cancellationToken);
            var vectors = new float[utterances.Count][];
            try
            {
                for (int i = 0; i < vectors.Length; i++) vectors[i] = Extract(utterances[i], cancellationToken);
                var memories = new ReadOnlyMemory<float>[vectors.Length];
                for (int i = 0; i < vectors.Length; i++) memories[i] = vectors[i];
                float[] average = EmbeddingMath.Average(memories);
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    enrollments.TryGetValue(speakerId, out var old);
                    enrollments[speakerId] = average;
                    if (old is not null) Array.Clear(old);
                }
                catch { Array.Clear(average); throw; }
            }
            finally
            {
                foreach (var vector in vectors)
                {
                    if (vector is not null) Array.Clear(vector);
                }
            }
        }
    }

    /// <summary>Finds the highest score; exact ties use ordinal ID order. A score equal to Threshold is accepted.
    /// Returns null ID below threshold or with no enrollments (score negative infinity in the latter case).
    /// Throws ArgumentException for invalid audio, OperationCanceledException on cancellation,
    /// or ObjectDisposedException after disposal.</summary>
    public override SpeakerMatch Identify(AudioBuffer audio, CancellationToken cancellationToken)
    {
        lock (sync)
        {
            Check(cancellationToken);
            var vector = Extract(audio, cancellationToken);
            try
            {
                return IdentifyEmbedding(vector, cancellationToken);
            }
            finally { Array.Clear(vector); }
        }
    }

    internal SpeakerMatch IdentifyEmbedding(ReadOnlyMemory<float> vector, CancellationToken cancellationToken)
    {
        lock (sync)
        {
            Check(cancellationToken);
            double best = double.NegativeInfinity;
            string? id = null;
            foreach (var pair in enrollments)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double score = Score(vector.Span, pair.Value, cancellationToken);
                if (score > best || (score == best && string.CompareOrdinal(pair.Key, id) < 0))
                {
                    best = score;
                    id = pair.Key;
                }
            }
            return new SpeakerMatch { SpeakerId = best >= Threshold ? id : null, Score = best };
        }
    }

    /// <summary>Scores two utterances without changing enrollments. Raw cosine is in [-1,1]; AS-norm is unbounded.
    /// Throws ArgumentException for invalid audio, OperationCanceledException on cancellation,
    /// or ObjectDisposedException after disposal. A score is not a probability.</summary>
    public override double Verify(AudioBuffer first, AudioBuffer second, CancellationToken cancellationToken)
    {
        lock (sync)
        {
            Check(cancellationToken);
            var a = Extract(first, cancellationToken);
            try
            {
                var b = Extract(second, cancellationToken);
                try { return Score(a, b, cancellationToken); }
                finally { Array.Clear(b); }
            }
            finally { Array.Clear(a); }
        }
    }

    /// <summary>Clears and removes an enrollment; returns false if absent. Throws ArgumentException for a blank ID,
    /// OperationCanceledException on cancellation or ObjectDisposedException after disposal.</summary>
    public override bool Remove(string speakerId, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(speakerId);
        lock (sync)
        {
            Check(cancellationToken);
            if (!enrollments.Remove(speakerId, out var vector)) return false;
            Array.Clear(vector);
            return true;
        }
    }

    /// <summary>Clears enrollment storage. Does not dispose the borrowed embedder; repeated calls are harmless.
    /// Clearing managed storage is best effort, not a guarantee against memory dumps or runtime copies.</summary>
    public void Dispose()
    {
        lock (sync)
        {
            foreach (var vector in enrollments.Values) Array.Clear(vector);
            enrollments.Clear();
            disposed = true;
        }
    }

    private float[] Extract(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Channels != 1 || audio.SampleRate != embedder.SampleRate || audio.FrameCount == 0)
            throw new ArgumentException("Nonempty mono audio at the embedder sample rate is required.", nameof(audio));
        var vector = new float[embedder.Dimension];
        try
        {
            embedder.Embed(audio, vector, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            EmbeddingMath.Normalize(vector, vector);
            return vector;
        }
        catch { Array.Clear(vector); throw; }
    }

    private double Score(ReadOnlySpan<float> first, ReadOnlySpan<float> second, CancellationToken cancellationToken)
        => normalizer is null ? EmbeddingMath.Cosine(first, second) : normalizer.Score(first, second, cancellationToken);

    private void Check(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
    }
}
