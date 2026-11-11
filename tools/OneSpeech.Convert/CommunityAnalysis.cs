using OneSpeech.Formats;

// Compares hypotheses about the community pipeline's aggregation against hook artifacts recorded from the released reference.
internal static class CommunityAnalysis
{
    internal static string Find(string dir, string suffix) => Path.GetFileName(Directory.GetFiles(dir, "*" + suffix).Single())!;
    private static NpyArray Load(string dir, string name) => NpzArchive.ParseNpy(File.ReadAllBytes(Path.Combine(dir, name)));

    internal static void Run(string dir)
    {
        var seg = Load(dir, Find(dir, "-segmentation.npy")); // [chunks, 589, 3]
        var count = Load(dir, Find(dir, "-speaker_counting.npy")); // [frames, 1]
        int chunks = seg.Shape[0], frames = seg.Shape[1], locals = seg.Shape[2], total = count.Shape[0];
        var distinct = new SortedSet<double>(seg.Values.Take(100000));
        Console.WriteLine($"chunks={chunks} frames={frames} total={total} distinct segmentation values: {string.Join(",", distinct.Take(8))} (n={distinct.Count})");
        var counts = new SortedDictionary<double, int>();
        foreach (double v in count.Values) counts[v] = counts.GetValueOrDefault(v) + 1;
        Console.WriteLine("reference count histogram: " + string.Join(" ", counts.Select(p => $"{p.Key}:{p.Value}")));

        foreach (string place in new[] { "round", "floor", "ceil" })
        foreach (bool hamming in new[] { false, true })
        foreach (string rounding in new[] { "round", "ceil", "floor" })
        {
            var sum = new double[total]; var weight = new double[total];
            for (int c = 0; c < chunks; c++)
            {
                double origin = c * 16000.0 / 270.0;
                int offset = place switch { "round" => (int)Math.Round(origin), "floor" => (int)Math.Floor(origin), _ => (int)Math.Ceiling(origin) };
                for (int f = 0; f < frames; f++)
                {
                    int i = offset + f;
                    if (i >= total) continue;
                    double v = 0; for (int k = 0; k < locals; k++) v += seg.Values[(c * frames + f) * locals + k];
                    double w = hamming ? 0.54 - 0.46 * Math.Cos(2 * Math.PI * f / (frames - 1)) : 1;
                    sum[i] += w * v; weight[i] += w;
                }
            }
            int match = 0, valid = 0;
            for (int i = 0; i < total; i++)
            {
                if (weight[i] == 0) continue;
                valid++;
                double avg = sum[i] / weight[i];
                double r = rounding switch { "round" => Math.Round(avg, MidpointRounding.ToEven), "ceil" => Math.Ceiling(avg), _ => Math.Floor(avg) };
                if (r == count.Values[i]) match++;
            }
            Console.WriteLine($"place={place} hamming={hamming} rounding={rounding}: {match}/{valid}");
        }
    }
}
