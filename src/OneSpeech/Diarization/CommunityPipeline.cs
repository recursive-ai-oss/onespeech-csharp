using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>Tunable constants of the community pipeline. Defaults are the publisher's configuration where it states one
/// (AHC threshold 0.6, F_A 0.07, F_B 0.8); the rest were set by comparing intermediate outputs with the released reference.</summary>
internal sealed record CommunityOptions
{
    internal double AhcThreshold { get; init; } = 0.6;
    internal double Fa { get; init; } = 0.07;
    internal double Fb { get; init; } = 0.8;
    internal double LoopP { get; init; } = 0.7;
    internal double Smoothing { get; init; } = 7;
    internal int MaxIterations { get; init; } = 20;
    internal double Epsilon { get; init; } = 1e-4;
    internal bool ReassignAll { get; init; }
    internal int MinClusterSize { get; init; }
    internal int PostMinClusterSize { get; init; } = 12;
    internal bool AhcInLdaSpace { get; init; } = true;
    internal bool HammingActivity { get; init; }
}

/// <summary>Result with the intermediate arrays that the comparison tooling inspects.</summary>
internal sealed class CommunityResult
{
    internal required int ChunkCount { get; init; }
    internal required byte[][] Masks { get; init; }          // [chunk][589 * 3]
    internal required int[] Counts { get; init; }            // speaker count per output frame
    internal required float[][] Embeddings { get; init; }    // [chunk * 3 + speaker], null when absent
    internal required int[] Assignment { get; init; }        // cluster per [chunk * 3 + speaker], -1 when absent
    internal required float[][] Centroids { get; init; }
    internal required byte[][] Activity { get; init; }       // [frame][cluster]
    internal required ImmutableArray<Turn> Turns { get; init; }
}

/// <summary>Community pipeline (Bredin 2023: local speakers per 10 s window, exclusive-speech embeddings, global clustering,
/// reconstruction by speaker count): 10 s windows every 1 s, powerset segmentation, WeSpeaker embeddings weighted by
/// overlap-free activity, PLDA + VBx clustering, then overlap-add reconstruction keeping the top-n clusters per frame,
/// where n is the rounded mean of the window speaker counts.</summary>
internal sealed class CommunityPipeline(PyannoteSegmenter segmenter, WeSpeakerEmbedder embedder, PldaTransform plda, CommunityOptions options)
{
    internal const int Window = 160000, Step = 16000, Frames = PyannoteSegmenter.Frames, FrameStep = 270, FrameWidth = 991, Locals = 3;

    internal static Action<string>? Diagnostics;

    internal static int ChunkCount(int samples) => samples <= Window ? 1 : (samples - Window + Step - 1) / Step + 1;

    // Output frame grid: the segmentation frame grid (hop 270 samples, width 991 samples) over the whole extent of the windows.
    internal static int TotalFrames(int chunks) => (int)Math.Round(((chunks - 1) * (double)Step + Window) / FrameStep) + 1;
    internal static int Offset(int chunk) => (int)Math.Round(chunk * (double)Step / FrameStep);

    internal CommunityResult Run(float[] audio, CancellationToken cancellationToken)
    {
        int chunks = ChunkCount(audio.Length);
        var masks = new byte[chunks][];
        var embeddings = new float[chunks * Locals][];
        var chunk = new float[Window];
        for (int c = 0; c < chunks; c++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Array.Clear(chunk);
            int start = c * Step, n = Math.Max(0, Math.Min(Window, audio.Length - start));
            Array.Copy(audio, start, chunk, 0, n);
            float[] lp = segmenter.GetFrameLogProbabilities(new AudioBuffer(chunk, 16000), cancellationToken);
            var mask = new byte[Frames * Locals];
            var combined = new int[Frames];
            for (int f = 0; f < Frames; f++)
            {
                int best = 0;
                for (int k = 1; k < 7; k++) if (lp[f * 7 + k] > lp[f * 7 + best]) best = k;
                int bits = PyannoteSegmenter.ClassMasks[best];
                combined[f] = bits;
                for (int s = 0; s < Locals; s++) mask[f * Locals + s] = (byte)(bits >> s & 1);
            }
            masks[c] = mask;
            // Frame features once per window; each local speaker only changes the pooling weights.
            float[]? frames = null; int steps = 0;
            var weights = new float[Frames];
            for (int s = 0; s < Locals; s++)
            {
                int active = 0, clean = 0;
                for (int f = 0; f < Frames; f++)
                {
                    bool on = (combined[f] >> s & 1) != 0;
                    if (!on) { weights[f] = 0; continue; }
                    active++;
                    bool alone = (combined[f] & (combined[f] - 1)) == 0;
                    weights[f] = alone ? 1 : 0; if (alone) clean++;
                }
                if (active == 0) continue;
                // Observed in the reference's embeddings: a speaker with no overlap-free frame is embedded from all their frames.
                if (clean == 0) for (int f = 0; f < Frames; f++) weights[f] = (combined[f] >> s & 1);
                frames ??= embedder.Frames(chunk, out steps, cancellationToken);
                float[] e = embedder.Project(WeSpeakerEmbedder.Pool(frames, steps, weights));
                bool finite = true; foreach (float v in e) if (!float.IsFinite(v)) finite = false;
                if (finite) embeddings[c * Locals + s] = e;
            }
        }
        return Finish(chunks, audio.Length, masks, embeddings);
    }

    /// <summary>Clustering and reconstruction from window masks (bytes [frame * 3 + speaker]) and per-speaker embeddings (null when absent).</summary>
    internal CommunityResult Finish(int chunks, int samples, byte[][] masks, float[][] embeddings)
    {
        int total = TotalFrames(chunks);
        var counts = new int[total];
        {
            var sum = new double[total]; var seen = new int[total];
            for (int c = 0; c < chunks; c++)
                for (int f = 0; f < Frames; f++)
                {
                    int i = Offset(c) + f; if (i >= total) continue;
                    int v = 0; for (int s = 0; s < Locals; s++) v += masks[c][f * Locals + s];
                    sum[i] += v; seen[i]++;
                }
            for (int i = 0; i < total; i++) counts[i] = seen[i] == 0 ? 0 : (int)Math.Round(sum[i] / seen[i], MidpointRounding.ToEven);
        }
        var (assignment, centroids) = Cluster(embeddings, chunks);
        int clusters = centroids.Length;
        var activity = new byte[total][];
        for (int i = 0; i < total; i++) activity[i] = new byte[clusters];
        if (clusters > 0)
        {
            var sum = new double[total * clusters]; var seen = new int[total];
            for (int c = 0; c < chunks; c++)
                for (int f = 0; f < Frames; f++)
                {
                    int i = Offset(c) + f; if (i >= total) continue;
                    seen[i]++;
                    double weight = options.HammingActivity ? 0.54 - 0.46 * Math.Cos(2 * Math.PI * f / (Frames - 1)) : 1;
                    for (int s = 0; s < Locals; s++)
                        if (assignment[c * Locals + s] >= 0 && masks[c][f * Locals + s] != 0) sum[i * clusters + assignment[c * Locals + s]] += weight;
                }
            // A cluster reached by two local speakers of one window counts once per window; the constrained assignment makes that impossible.
            var order = new int[clusters];
            for (int i = 0; i < total; i++)
            {
                if (seen[i] == 0) continue;
                for (int k = 0; k < clusters; k++) order[k] = k;
                int n = Math.Min(counts[i], clusters);
                var row = sum.AsSpan(i * clusters, clusters);
                // Stable top-n by mean activation; ties go to the lower cluster index.
                for (int a = 0; a < n; a++)
                {
                    int best = a;
                    for (int b = a + 1; b < clusters; b++) if (row[order[b]] > row[order[best]]) best = b;
                    (order[a], order[best]) = (order[best], order[a]);
                    activity[i][order[a]] = 1;
                }
            }
        }
        return new CommunityResult
        {
            ChunkCount = chunks, Masks = masks, Counts = counts, Embeddings = embeddings, Assignment = assignment, Centroids = centroids,
            Activity = activity, Turns = Turns(activity, clusters, samples),
        };
    }

    private (int[] Assignment, float[][] Centroids) Cluster(float[][] embeddings, int chunks)
    {
        var assignment = new int[embeddings.Length];
        Array.Fill(assignment, -1);
        var index = new List<int>();
        for (int i = 0; i < embeddings.Length; i++) if (embeddings[i] is not null) index.Add(i);
        if (index.Count == 0) return (assignment, []);
        if (index.Count == 1) { assignment[index[0]] = 0; return (assignment, [embeddings[index[0]]]); }
        var unit = new float[index.Count][];
        for (int i = 0; i < unit.Length; i++) unit[i] = Normalize(embeddings[index[i]]);
        float[][] ahc = unit;
        if (options.AhcInLdaSpace) ahc = index.Select(i => plda.Whitened(embeddings[i]).Select(v => (float)v).ToArray()).ToArray();
        int[] init = CentroidLinkage.Cluster(ahc, options.AhcThreshold);
        if (options.MinClusterSize > 1) init = MergeSmall(ahc, init, options.MinClusterSize);
        int initial = init.Max() + 1;
        Diagnostics?.Invoke("AHC sizes: " + string.Join(",", Enumerable.Range(0, initial).Select(k => init.Count(l => l == k))));
        int[] labels = init;
        if (initial > 1)
        {
            var rho = new double[unit.Length][];
            double[] phi = plda.Psi;
            for (int i = 0; i < rho.Length; i++)
            {
                double[] z = plda.Apply(embeddings[index[i]]);
                rho[i] = new double[z.Length];
                for (int d = 0; d < z.Length; d++) rho[i][d] = z[d] * Math.Sqrt(phi[d]);
            }
            var vbx = VbxClustering.Run(rho, phi, init, initial, options.Fa, options.Fb, options.LoopP, options.Smoothing, options.MaxIterations, options.Epsilon);
            var keep = new List<int>();
            for (int s = 0; s < initial; s++) if (vbx.Priors[s] > 1e-7) keep.Add(s);
            if (keep.Count == 0) keep.Add(0);
            labels = new int[unit.Length];
            for (int i = 0; i < labels.Length; i++)
            {
                int best = 0;
                for (int k = 1; k < keep.Count; k++) if (vbx.Responsibilities[i][keep[k]] > vbx.Responsibilities[i][keep[best]]) best = k;
                labels[i] = best;
            }
        }
        if (options.PostMinClusterSize > 1) labels = MergeSmall(unit, labels, options.PostMinClusterSize);
        Diagnostics?.Invoke("final sizes: " + string.Join(",", labels.GroupBy(l => l).OrderBy(g => g.Key).Select(g => g.Count())));
        // Renumber so every cluster has a member, then take centroids as the mean embedding of each cluster.
        var map = new Dictionary<int, int>();
        for (int i = 0; i < labels.Length; i++) if (!map.ContainsKey(labels[i])) map[labels[i]] = map.Count;
        int clusters = map.Count;
        var centroids = new float[clusters][];
        var counts = new int[clusters];
        for (int k = 0; k < clusters; k++) centroids[k] = new float[WeSpeakerEmbedder.Dimension];
        for (int i = 0; i < labels.Length; i++)
        {
            int k = map[labels[i]]; counts[k]++;
            for (int d = 0; d < centroids[k].Length; d++) centroids[k][d] += embeddings[index[i]][d];
            assignment[index[i]] = k;
        }
        for (int k = 0; k < clusters; k++) for (int d = 0; d < centroids[k].Length; d++) centroids[k][d] /= counts[k];
        // Two local speakers of one window are different people: where the clustering gave them one cluster, or when
        // asked to reassign everything, choose the best injective assignment by cosine similarity to the centroids.
        var unitCentroids = centroids.Select(Normalize).ToArray();
        for (int c = 0; c < chunks; c++)
        {
            var present = new List<int>();
            for (int s = 0; s < Locals; s++) if (assignment[c * Locals + s] >= 0) present.Add(s);
            bool clash = present.Select(s => assignment[c * Locals + s]).Distinct().Count() != present.Count;
            if (present.Count == 0 || !(clash || options.ReassignAll) || clusters < present.Count) continue;
            var best = Injective(present.Select(s => Normalize(embeddings[c * Locals + s]!)).ToArray(), unitCentroids);
            for (int p = 0; p < present.Count; p++) assignment[c * Locals + present[p]] = best[p];
        }
        return (assignment, centroids);
    }

    // Clusters smaller than minimum are merged into the nearest larger cluster (by centroid distance); labels are renumbered.
    private static int[] MergeSmall(float[][] points, int[] labels, int minimum)
    {
        int k = labels.Max() + 1;
        var size = new int[k]; foreach (int l in labels) size[l]++;
        var centre = new double[k][];
        for (int c = 0; c < k; c++) centre[c] = new double[points[0].Length];
        for (int i = 0; i < points.Length; i++) for (int d = 0; d < centre[labels[i]].Length; d++) centre[labels[i]][d] += points[i][d] / size[labels[i]];
        var target = new int[k];
        bool anyLarge = size.Any(x => x >= minimum);
        for (int c = 0; c < k; c++)
        {
            target[c] = c;
            if (size[c] >= minimum || !anyLarge) continue;
            double best = double.PositiveInfinity;
            for (int o = 0; o < k; o++)
            {
                if (size[o] < minimum) continue;
                double dist = 0; for (int d = 0; d < centre[c].Length; d++) dist += (centre[c][d] - centre[o][d]) * (centre[c][d] - centre[o][d]);
                if (dist < best) { best = dist; target[c] = o; }
            }
        }
        var map = new Dictionary<int, int>(); var result = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++) { int t = target[labels[i]]; if (!map.TryGetValue(t, out int id)) map[t] = id = map.Count; result[i] = id; }
        return result;
    }

    // At most three local speakers: exhaustive injective assignment is bounded by clusters^3.
    internal static int[] Injective(float[][] points, float[][] centers)
    {
        int[] current = new int[points.Length], best = new int[points.Length]; double bestScore = double.NegativeInfinity;
        void Visit(int depth, ulong used, double score)
        {
            if (depth == points.Length) { if (score > bestScore) { bestScore = score; current.CopyTo(best, 0); } return; }
            for (int c = 0; c < centers.Length; c++)
            {
                if (c < 64 && (used & 1UL << c) != 0) continue;
                double dot = 0; for (int d = 0; d < centers[c].Length; d++) dot += points[depth][d] * (double)centers[c][d];
                current[depth] = c; Visit(depth + 1, c < 64 ? used | 1UL << c : used, score + dot);
            }
        }
        Visit(0, 0, 0); return best;
    }

    internal static float[] Normalize(float[] v)
    {
        double n = 0; foreach (float x in v) n += (double)x * x;
        n = Math.Sqrt(n); var r = new float[v.Length];
        if (n > 0) for (int i = 0; i < v.Length; i++) r[i] = (float)(v[i] / n);
        return r;
    }

    // Binarization at frame centres: a turn starts at the centre of its first active frame and ends at the centre of the
    // first inactive frame after it (observed in the reference's turn boundaries, which are multiples of 270 samples).
    private static ImmutableArray<Turn> Turns(byte[][] activity, int clusters, int samples)
    {
        var turns = ImmutableArray.CreateBuilder<Turn>();
        double Seconds(int frame) => Math.Min(samples, frame * (double)FrameStep + (FrameWidth - 1) / 2.0 + 0.5) / 16000.0;
        for (int k = 0; k < clusters; k++)
        {
            int start = -1;
            for (int i = 0; i <= activity.Length; i++)
            {
                bool on = i < activity.Length && activity[i][k] != 0;
                if (on && start < 0) start = i;
                if (!on && start >= 0)
                {
                    double a = Seconds(start), b = Seconds(i);
                    if (b > a) turns.Add(new Turn { Speaker = new SpeakerLabel { Id = "speaker-" + k }, Start = TimeSpan.FromSeconds(a), End = TimeSpan.FromSeconds(b), Text = "" });
                    start = -1;
                }
            }
        }
        return turns.ToImmutable().Sort((x, y) => x.Start.CompareTo(y.Start));
    }
}
