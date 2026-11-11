using OneSpeech.Diarization;
using OneSpeech.Formats;

// Runs clustering and reconstruction on the reference's own segmentation and embeddings, then compares the outcome
// with the reference's counts, centroids and frame-level diarization. Isolates the clustering hypotheses from the networks.
internal static class CommunityCluster
{
    private static NpyArray Load(string dir, string name) => NpzArchive.ParseNpy(File.ReadAllBytes(Path.Combine(dir, name)));
    private static double Env(string name, double fallback) => double.TryParse(Environment.GetEnvironmentVariable(name), System.Globalization.CultureInfo.InvariantCulture, out double v) ? v : fallback;

    internal static void Run(string snapshot, string dir)
    {
        var seg = Load(dir, CommunityCompare.Find(dir, "-segmentation.npy"));
        var count = Load(dir, CommunityCompare.Find(dir, "-speaker_counting.npy"));
        var disc = Load(dir, CommunityCompare.Find(dir, "-discrete_diarization.npy"));
        var emb = Load(dir, Directory.GetFiles(dir, "*-embeddings.npy").Select(Path.GetFileName).Order(StringComparer.Ordinal).Last()!);
        var cent = Load(dir, "speaker_embeddings.npy");
        int chunks = seg.Shape[0];
        var masks = new byte[chunks][]; var embeddings = new float[chunks * 3][];
        for (int c = 0; c < chunks; c++)
        {
            masks[c] = new byte[589 * 3];
            var active = new int[3];
            for (int f = 0; f < 589; f++) for (int s = 0; s < 3; s++) { byte v = (byte)(seg.Values[(c * 589 + f) * 3 + s] > 0.5 ? 1 : 0); masks[c][f * 3 + s] = v; active[s] += v; }
            for (int s = 0; s < 3; s++)
            {
                if (active[s] == 0) continue;
                var e = new float[256]; bool finite = true;
                for (int i = 0; i < 256; i++) { e[i] = (float)emb.Values[(c * 3 + s) * 256 + i]; if (!float.IsFinite(e[i])) finite = false; }
                if (finite) embeddings[c * 3 + s] = e;
            }
        }
        PldaTransform.TransposeRotation = Env("TR", 1) != 0;
        var plda = PldaTransform.Load(Path.Combine(snapshot, "plda/plda.npz"), Path.Combine(snapshot, "plda/xvec_transform.npz"));
        string? grid = Environment.GetEnvironmentVariable("GRID");
        if (Environment.GetEnvironmentVariable("DIAG") is not null) CommunityPipeline.Diagnostics = Console.WriteLine;
        var configs = new List<CommunityOptions>();
        if (grid is null) configs.Add(Options());
        else foreach (double loop in new[] { 0.3, 0.5, 0.7, 0.8, 0.9, 0.99 }) foreach (int mcs in new[] { 0, 5, 12, 15 }) foreach (double sm in new[] { 5.0, 7.0, 10.0 })
            configs.Add(Options() with { LoopP = loop, MinClusterSize = mcs, Smoothing = sm });
        foreach (var options in configs)
        {
        if (grid is not null) Console.Write($"LOOP={options.LoopP} MCS={options.MinClusterSize} SM={options.Smoothing}: ");
        var pipeline = new CommunityPipeline(null!, null!, plda, options);
        var result = pipeline.Finish(chunks, (chunks - 1) * 16000 + 160000, masks, embeddings);
        int total = Math.Min(count.Shape[0], result.Counts.Length);
        int same = 0; for (int i = 0; i < total; i++) if (result.Counts[i] == (int)count.Values[i]) same++;
        Console.WriteLine($"frames ours={result.Counts.Length} ref={count.Shape[0]}; counts equal {same}/{total}");
        int refK = cent.Shape[0], ourK = result.Centroids.Length;
        Console.WriteLine($"clusters ours={ourK} ref centroids={refK} ref activity columns={disc.Shape[1]}");
        for (int r = 0; r < refK; r++)
        {
            double best = -2; int bi = -1;
            for (int k = 0; k < ourK; k++)
            {
                double dot = 0, na = 0, nb = 0;
                for (int d = 0; d < 256; d++) { double a = result.Centroids[k][d], b = cent.Values[r * 256 + d]; dot += a * b; na += a * a; nb += b * b; }
                double cos = dot / Math.Sqrt(na * nb); if (cos > best) { best = cos; bi = k; }
            }
            var all = new List<string>();
            for (int k = 0; k < ourK; k++)
            {
                double dot = 0, na = 0, nb = 0;
                for (int d = 0; d < 256; d++) { double a = result.Centroids[k][d], b = cent.Values[r * 256 + d]; dot += a * b; na += a * a; nb += b * b; }
                all.Add((dot / Math.Sqrt(na * nb)).ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
            }
            Console.WriteLine($"  ref centroid {r}: best ours {bi} cosine {best:F6}; all ours: {string.Join(" ", all)}");
        }
        if (Environment.GetEnvironmentVariable("MEMBERS") is not null)
        {
            // Per cluster: size, chunk range, and how close the members are to the nearest reference centroid.
            for (int k = 0; k < ourK; k++)
            {
                var members = Enumerable.Range(0, result.Assignment.Length).Where(i => result.Assignment[i] == k && embeddings[i] is not null).ToList();
                var best = new List<double>(); var chunksOf = members.Select(i => i / 3).ToList();
                foreach (int i in members)
                {
                    double top = -2;
                    for (int r = 0; r < refK; r++)
                    {
                        double dot = 0, na = 0, nb = 0;
                        for (int d = 0; d < 256; d++) { double a = embeddings[i]![d], b = cent.Values[r * 256 + d]; dot += a * b; na += a * a; nb += b * b; }
                        top = Math.Max(top, dot / Math.Sqrt(na * nb));
                    }
                    best.Add(top);
                }
                best.Sort();
                Console.WriteLine($"  cluster {k}: n={members.Count} chunks {chunksOf.Min()}..{chunksOf.Max()} (median {chunksOf.Order().ElementAt(chunksOf.Count / 2)}) best-ref-cos p10={best[best.Count / 10]:F3} median={best[best.Count / 2]:F3} p90={best[best.Count * 9 / 10]:F3}");
            }
        }
        // Frame agreement under the best one-to-one pairing of columns (greedy by overlap).
        int cols = disc.Shape[1], n = Math.Min(total, disc.Shape[0]);
        var used = new bool[ourK]; long agree = 0, refOn = 0, oursOn = 0, both = 0;
        for (int r = 0; r < cols; r++)
        {
            int bi = -1; long bo = -1;
            for (int k = 0; k < ourK; k++)
            {
                if (used[k]) continue;
                long o = 0; for (int i = 0; i < n; i++) if (disc.Values[i * cols + r] > 0.5 && result.Activity[i][k] != 0) o++;
                if (o > bo) { bo = o; bi = k; }
            }
            if (bi >= 0) used[bi] = true;
            for (int i = 0; i < n; i++)
            {
                bool a = disc.Values[i * cols + r] > 0.5, b = bi >= 0 && result.Activity[i][bi] != 0;
                if (a == b) agree++; if (a) refOn++; if (b) oursOn++; if (a && b) both++;
            }
        }
        Console.WriteLine($"discrete diarization: agree {agree}/{(long)n * cols} ({100.0 * agree / ((long)n * cols):F3}%), ref on {refOn}, ours on {oursOn}, both {both}");
        }
    }

    private static CommunityOptions Options() => new()
    {
        AhcThreshold = Env("AHC", 0.6), LoopP = Env("LOOP", 0.7), Smoothing = Env("SMOOTH", 7), Fa = Env("FA", 0.07), Fb = Env("FB", 3.2),
        MaxIterations = (int)Env("ITERS", 20), Epsilon = Env("EPS", 1e-4), ReassignAll = Env("REASSIGN", 0) != 0, MinClusterSize = (int)Env("MCS", 0), PostMinClusterSize = (int)Env("PMCS", 12), AhcInLdaSpace = Env("LDA", 1) != 0, MinTrainRatio = Env("MINTRAIN", 0.2), HammingActivity = Env("HAM", 0) != 0,
    };
}
