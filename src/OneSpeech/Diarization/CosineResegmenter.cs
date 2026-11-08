namespace OneSpeech.Diarization;

/// <summary>Cosine-centroid resegmentation with a switch penalty, using existing speech embeddings only.
/// Does not detect overlap, add speakers or change activity boundaries. Centroids and paths are local to each call; nothing is persisted.</summary>
public sealed class CosineResegmenter : EmbeddingResegmenter
{
    /// <summary>Creates a deterministic smoother. SwitchPenalty defaults to 0.1 cosine-distance units per label change (range 0–10).
    /// StableRadius defaults to 2 windows (range 0–16): a centroid uses windows whose neighboring initial labels agree, falling back to all its windows if none qualify.
    /// Throws ArgumentOutOfRangeException for invalid parameters. This is an algorithmic setting, not a learned model.</summary>
    public CosineResegmenter(double switchPenalty = 0.1, int stableRadius = 2)
    {
        if (!double.IsFinite(switchPenalty) || switchPenalty < 0 || switchPenalty > 10) throw new ArgumentOutOfRangeException(nameof(switchPenalty));
        if (stableRadius < 0 || stableRadius > 16) throw new ArgumentOutOfRangeException(nameof(stableRadius));
        SwitchPenalty = switchPenalty; StableRadius = stableRadius;
    }
    /// <summary>Nonnegative cost of changing speakers between neighboring windows, in cosine-distance units.</summary>
    public double SwitchPenalty { get; }
    /// <summary>Number of neighbors on either side required to share an initial label for centroid support.</summary>
    public int StableRadius { get; }

    /// <summary>Returns one existing speaker label per window, possibly removing a speaker. Empty input returns empty.
    /// Windows must be ordered by start, with positive intervals, finite unit embeddings of one dimension (1–4096), nonnegative labels and at most 32 distinct labels.
    /// At most 1,000,000 windows are accepted; throws ArgumentException for invalid input or larger counts.
    /// A gap between windows resets smoothing. Ties prefer staying, then the lowest label. A centroid norm below 1e-12 returns initial labels unchanged.
    /// Cancellation is checked per window and throws OperationCanceledException when requested.
    /// Extra storage is O(NK + KD) bytes for N windows, K speakers and D dimensions; no PCM is retained.</summary>
    public override int[] Resegment(IReadOnlyList<EmbeddingWindow> windows, ReadOnlySpan<int> initialLabels, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(windows); cancellationToken.ThrowIfCancellationRequested();
        int n = windows.Count;
        if (n != initialLabels.Length || n > 1_000_000) throw new ArgumentException("Resegmentation requires matching labels and at most 1,000,000 windows.");
        if (n == 0) return [];
        if (windows[0] is null) throw new ArgumentException("A window is null.", nameof(windows));
        int dimension = windows[0].Embedding.Length;
        if (dimension < 1 || dimension > 4096) throw new ArgumentException("Embedding dimension must be 1–4096.");
        var ids = new SortedSet<int>();
        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var w = windows[i];
            if (w is null || w.Embedding.Length != dimension) throw new ArgumentException("Null window or inconsistent embedding dimension.", nameof(windows));
            double norm = 0;
            foreach (float value in w.Embedding.Span) { if (!float.IsFinite(value)) throw new ArgumentException("Nonfinite embedding."); norm += (double)value * value; }
            if (w.Start < TimeSpan.Zero || w.End <= w.Start || i > 0 && w.Start < windows[i - 1].Start || w.Embedding.Length != dimension || Math.Abs(norm - 1) > 0.001 || initialLabels[i] < 0)
                throw new ArgumentException("Expected ordered positive windows, unit embeddings and nonnegative labels.");
            ids.Add(initialLabels[i]);
            if (ids.Count > 32) throw new ArgumentException("At most 32 speakers are supported.");
        }
        var labels = ids.ToArray(); int k = labels.Length;
        var map = new Dictionary<int, int>(); for (int s = 0; s < k; s++) map.Add(labels[s], s);
        var all = new double[k][]; var stable = new double[k][]; var supports = new int[k];
        for (int s = 0; s < k; s++) { all[s] = new double[dimension]; stable[s] = new double[dimension]; }
        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested(); int s = map[initialLabels[i]];
            bool interior = i >= StableRadius && i + StableRadius < n;
            for (int j = Math.Max(0, i - StableRadius); interior && j <= i + StableRadius; j++)
                if (initialLabels[j] != initialLabels[i] || j > 0 && windows[j].Start > windows[j - 1].End) interior = false;
            for (int d = 0; d < dimension; d++) { double v = windows[i].Embedding.Span[d]; all[s][d] += v; if (interior) stable[s][d] += v; }
            if (interior) supports[s]++;
        }
        var centers = new double[k][];
        for (int s = 0; s < k; s++)
        {
            centers[s] = supports[s] == 0 ? all[s] : stable[s];
            double norm = Math.Sqrt(centers[s].Sum(v => v * v));
            // An antipodal cluster has no useful mean; leave the initial labeling unchanged.
            if (norm < 1e-12) return initialLabels.ToArray();
            for (int d = 0; d < dimension; d++) centers[s][d] /= norm;
        }
        var back = new byte[checked(n * k)]; var previous = new double[k]; var current = new double[k]; var result = new int[n];
        int first = 0;
        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            bool reset = i == 0 || windows[i].Start > windows[i - 1].End;
            if (reset && i > 0) { Trace(i - 1, first); first = i; }
            for (int s = 0; s < k; s++)
            {
                int best = s; double cost = reset ? 0 : previous[s];
                if (!reset) for (int t = 0; t < k; t++) if (t != s && previous[t] + SwitchPenalty < cost) { cost = previous[t] + SwitchPenalty; best = t; }
                double dot = 0; for (int d = 0; d < dimension; d++) dot += windows[i].Embedding.Span[d] * centers[s][d];
                current[s] = cost + 1 - Math.Clamp(dot, -1, 1); back[i * k + s] = (byte)best;
            }
            (previous, current) = (current, previous);
        }
        Trace(n - 1, first); return result;

        void Trace(int last, int start)
        {
            int best = 0; for (int s = 1; s < k; s++) if (previous[s] < previous[best]) best = s;
            for (int i = last; i >= start; i--) { cancellationToken.ThrowIfCancellationRequested(); result[i] = labels[best]; best = back[i * k + best]; }
        }
    }
}
