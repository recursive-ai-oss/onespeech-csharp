namespace OneSpeech.Diarization;

// Park et al. 2020 §III: cosine, p-neighbor graph, D-A, NME, smallest eigenvectors, k-means.
// Dense work is limited to a deterministic landmark sample; all windows are then assigned by cosine centroids.
internal static class EmbeddingClustering
{
    internal static void Normalize(Span<float> vector)
    {
        double norm = 0;
        foreach (float value in vector) { if (!float.IsFinite(value)) throw new InvalidOperationException("Embedder returned nonfinite values."); norm += (double)value * value; }
        if (norm <= 0) throw new InvalidOperationException("Embedder returned a zero vector.");
        norm = Math.Sqrt(norm);
        for (int i = 0; i < vector.Length; i++) vector[i] = (float)(vector[i] / norm);
    }

    internal static int[] Cluster(IReadOnlyList<EmbeddingWindow> windows, ClusteringOptions options, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        int count = windows.Count;
        if (count == 0) return [];
        if (count < (options.NumSpeakers ?? options.MinSpeakers))
            throw new CapabilityNotSupportedException("The requested speaker count exceeds the available speech embedding windows.");
        int n = Math.Min(count, options.MaxLandmarks), dimension = windows[0].Embedding.Length;
        var indices = new int[n]; var points = new float[n][];
        for (int i = 0; i < n; i++) { indices[i] = n == 1 ? 0 : (int)((long)i * (count - 1) / (n - 1)); points[i] = windows[indices[i]].Embedding.ToArray(); }
        var cosine = new double[n, n];
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) cosine[i, j] = cosine[j, i] = Dot(points[i], points[j]);
        int[]? labels = options.Method == ClusteringMethod.Spectral ? Spectral(cosine, options, token) : null;
        labels ??= Agglomerative(cosine, options, token);
        int k = labels.Max() + 1;
        var centers = new float[k][];
        for (int i = 0; i < k; i++) centers[i] = new float[dimension];
        for (int i = 0; i < n; i++) for (int d = 0; d < dimension; d++) centers[labels[i]][d] += points[i][d];
        foreach (var center in centers)
        {
            double norm = Math.Sqrt(Dot(center, center));
            if (norm > 0) for (int d = 0; d < dimension; d++) center[d] = (float)(center[d] / norm);
        }
        var result = new int[count];
        for (int i = 0; i < count; i++)
        {
            token.ThrowIfCancellationRequested();
            double best = double.NegativeInfinity;
            for (int j = 0; j < k; j++) { double score = Dot(windows[i].Embedding.Span, centers[j]); if (score > best) { best = score; result[i] = j; } }
        }
        // Preserve landmark labels, including exact-count splits of indistinguishable embeddings.
        for (int i = 0; i < n; i++) result[indices[i]] = labels[i];
        return Canonical(result);
    }

    private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    { double sum = 0; for (int d = 0; d < a.Length; d++) sum += (double)a[d] * b[d]; return sum; }

    private static int[]? Spectral(double[,] cosine, ClusteringOptions options, CancellationToken token)
    {
        int n = cosine.GetLength(0);
        if (n == 1) return [0];
        if (options.NumSpeakers == 1 || options.MaxSpeakers == 1) return new int[n];
        bool identical = true;
        for (int i = 0; i < n && identical; i++) for (int j = 0; j < n && identical; j++) identical = cosine[i, j] >= 1 - 1e-6;
        if (identical) return Agglomerative(cosine, options, token);
        var ranks = new int[n][];
        for (int i = 0; i < n; i++) { int row = i; ranks[i] = Enumerable.Range(0, n).OrderByDescending(j => cosine[row, j]).ThenBy(j => j).ToArray(); }
        double bestRatio = double.PositiveInfinity; double[,]? bestVectors = null; int bestK = 1;
        int maximumP = Math.Max(2, n / 4);
        // Log-spaced p search, plus its upper endpoint, bounds dense decompositions on long recordings.
        var candidates = new SortedSet<int> { Math.Min(n, maximumP) };
        for (int p = 2; p <= maximumP; p *= 2) candidates.Add(p);
        foreach (int p in candidates)
        {
            token.ThrowIfCancellationRequested();
            var adjacency = new double[n, n];
            for (int i = 0; i < n; i++) for (int j = 0; j < p; j++) { int other = ranks[i][j]; adjacency[i, other] += 0.5; adjacency[other, i] += 0.5; }
            var laplacian = new double[n, n];
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { laplacian[i, i] += adjacency[i, j]; laplacian[i, j] -= adjacency[i, j]; }
            if (!Eigen(laplacian, token, out var values, out var vectors)) return null;
            int minimum = Math.Min(options.MinSpeakers, n), maximum = Math.Min(options.MaxSpeakers, n - 1);
            int k = options.NumSpeakers ?? minimum; double gap = -1;
            for (int c = minimum; c <= maximum; c++)
            {
                double current = values[c] - values[c - 1];
                if (current > gap) { gap = current; if (options.NumSpeakers is null) k = c; }
            }
            if (options.NumSpeakers is {} fixedCount && fixedCount < n) gap = values[fixedCount] - values[fixedCount - 1];
            if (k == n) gap = values[^1];
            double ratio = p * (Math.Max(0, values[^1]) + 1e-10) / Math.Max(1e-10, gap);
            if (ratio < bestRatio) { bestRatio = ratio; bestVectors = vectors; bestK = k; }
        }
        if (bestVectors is null) return null;
        var features = new double[n][];
        for (int i = 0; i < n; i++) { features[i] = new double[bestK]; for (int j = 0; j < bestK; j++) features[i][j] = bestVectors[i, j]; }
        return KMeans(features, bestK, token);
    }

    private static bool Eigen(double[,] a, CancellationToken token, out double[] values, out double[,] vectors)
    {
        int n = a.GetLength(0); var v = new double[n, n];
        for (int i = 0; i < n; i++) v[i, i] = 1;
        bool converged = false;
        for (int sweep = 0; sweep < 60; sweep++)
        {
            token.ThrowIfCancellationRequested();
            double maximum = 0, scale = 1;
            for (int i = 0; i < n; i++) { scale = Math.Max(scale, Math.Abs(a[i, i])); for (int j = i + 1; j < n; j++) maximum = Math.Max(maximum, Math.Abs(a[i, j])); }
            if (maximum <= 1e-10 * scale) { converged = true; break; }
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++)
            {
                double entry = a[p, q]; if (Math.Abs(entry) < 1e-12 * scale) continue;
                double tau = (a[q, q] - a[p, p]) / (2 * entry);
                double t = Math.CopySign(1, tau) / (Math.Abs(tau) + Math.Sqrt(1 + tau * tau));
                double c = 1 / Math.Sqrt(1 + t * t), s = t * c;
                a[p, p] -= t * entry; a[q, q] += t * entry; a[p, q] = a[q, p] = 0;
                for (int r = 0; r < n; r++)
                {
                    if (r != p && r != q) { double x = a[r, p], y = a[r, q]; a[r, p] = a[p, r] = c * x - s * y; a[r, q] = a[q, r] = s * x + c * y; }
                    double vx = v[r, p], vy = v[r, q]; v[r, p] = c * vx - s * vy; v[r, q] = s * vx + c * vy;
                }
            }
        }
        var order = Enumerable.Range(0, n).OrderBy(i => a[i, i]).ThenBy(i => i).ToArray();
        values = new double[n]; vectors = new double[n, n];
        for (int j = 0; j < n; j++) { values[j] = a[order[j], order[j]]; for (int i = 0; i < n; i++) vectors[i, j] = v[i, order[j]]; }
        return converged;
    }

    private static int[] KMeans(double[][] x, int k, CancellationToken token)
    {
        int n = x.Length, dimension = x[0].Length;
        var centers = new double[k][]; var chosen = new bool[n]; centers[0] = (double[])x[0].Clone(); chosen[0] = true;
        double Distance(double[] a, double[] b) { double sum = 0; for (int d = 0; d < dimension; d++) { double diff = a[d] - b[d]; sum += diff * diff; } return sum; }
        for (int c = 1; c < k; c++)
        {
            int index = -1; double farthest = -1;
            for (int i = 0; i < n; i++) if (!chosen[i]) { double nearest = double.PositiveInfinity; for (int j = 0; j < c; j++) nearest = Math.Min(nearest, Distance(x[i], centers[j])); if (nearest > farthest) { farthest = nearest; index = i; } }
            chosen[index] = true; centers[c] = (double[])x[index].Clone();
        }
        var labels = Enumerable.Repeat(-1, n).ToArray();
        for (int iteration = 0; iteration < 100; iteration++)
        {
            token.ThrowIfCancellationRequested(); bool changed = false; var counts = new int[k];
            for (int i = 0; i < n; i++) { int best = 0; double minimum = double.PositiveInfinity; for (int j = 0; j < k; j++) { double distance = Distance(x[i], centers[j]); if (distance < minimum) { minimum = distance; best = j; } } changed |= labels[i] != best; labels[i] = best; counts[best]++; }
            for (int j = 0; j < k; j++) if (counts[j] == 0)
            {
                int donor = -1; double farthest = -1;
                for (int i = 0; i < n; i++) if (counts[labels[i]] > 1) { double distance = Distance(x[i], centers[labels[i]]); if (distance > farthest) { farthest = distance; donor = i; } }
                counts[labels[donor]]--; labels[donor] = j; counts[j]++; changed = true;
            }
            for (int j = 0; j < k; j++) Array.Clear(centers[j]);
            for (int i = 0; i < n; i++) for (int d = 0; d < dimension; d++) centers[labels[i]][d] += x[i][d];
            for (int j = 0; j < k; j++) for (int d = 0; d < dimension; d++) centers[j][d] /= counts[j];
            if (!changed) break;
        }
        return Canonical(labels);
    }

    // Average-link cosine AHC, matching the cosine initialization described by Landini et al. §3.2.
    private static int[] Agglomerative(double[,] cosine, ClusteringOptions options, CancellationToken token)
    {
        int n = cosine.GetLength(0); var active = Enumerable.Repeat(true, n).ToArray(); var sizes = Enumerable.Repeat(1, n).ToArray(); var labels = Enumerable.Range(0, n).ToArray(); int remaining = n;
        int minimum = options.NumSpeakers ?? options.MinSpeakers;
        while (remaining > minimum)
        {
            token.ThrowIfCancellationRequested(); int a = -1, b = -1; double maximum = double.NegativeInfinity;
            for (int i = 0; i < n; i++) if (active[i]) for (int j = i + 1; j < n; j++) if (active[j] && cosine[i, j] > maximum) { maximum = cosine[i, j]; a = i; b = j; }
            if (options.NumSpeakers is null && remaining <= options.MaxSpeakers && maximum < options.AgglomerativeThreshold) break;
            for (int j = 0; j < n; j++) if (active[j] && j != a && j != b) cosine[a, j] = cosine[j, a] = (cosine[a, j] * sizes[a] + cosine[b, j] * sizes[b]) / (sizes[a] + sizes[b]);
            sizes[a] += sizes[b]; active[b] = false; remaining--;
            for (int i = 0; i < n; i++) if (labels[i] == b) labels[i] = a;
        }
        return Canonical(labels);
    }

    internal static int[] Canonical(int[] labels)
    { var map = new Dictionary<int, int>(); for (int i = 0; i < labels.Length; i++) { if (!map.TryGetValue(labels[i], out int value)) { value = map.Count; map.Add(labels[i], value); } labels[i] = value; } return labels; }
}
