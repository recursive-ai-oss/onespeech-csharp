using System.Buffers.Binary;
using System.Globalization;
using OneSpeech.Speakers;

internal static class SpeakerCalibration
{
    internal static void Run(string manifest, string embeddings)
    {
        var trials = SpeakerVerification.ReadTrials(manifest);
        var speakers = trials.SelectMany(t => new[] { t.First.Split('-')[0], t.Second.Split('-')[0] }).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        var dev = speakers.Take(speakers.Length / 3).ToHashSet(StringComparer.Ordinal);
        Console.WriteLine("development-speakers=" + string.Join(',', dev.Order(StringComparer.Ordinal)));
        var vectors = new Dictionary<string, float[]>(StringComparer.Ordinal);
        float[] Load(string id)
        {
            if (vectors.TryGetValue(id, out var value)) return value;
            byte[] bytes = File.ReadAllBytes(Path.Combine(embeddings, id + ".f32"));
            value = new float[bytes.Length / 4];
            for (int i = 0; i < value.Length; i++) value[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
            vectors.Add(id, value); return value;
        }
        var scores = trials.Where(t => dev.Contains(t.First.Split('-')[0]) && dev.Contains(t.Second.Split('-')[0]))
            .Select(t => (t.Same, Value: EmbeddingMath.Cosine(Load(t.First), Load(t.Second)))).ToArray();
        int positives = scores.Count(s => s.Same), negatives = scores.Length - positives;
        double eerGap = double.PositiveInfinity, eerThreshold = 1, dcf = double.PositiveInfinity, dcfThreshold = 1;
        // Descending candidates make ties prefer the stricter threshold. Acceptance is inclusive.
        foreach (double threshold in scores.Select(s => s.Value).Distinct().OrderDescending())
        {
            double miss = (double)scores.Count(s => s.Same && s.Value < threshold) / positives;
            double fa = (double)scores.Count(s => !s.Same && s.Value >= threshold) / negatives;
            double gap = Math.Abs(miss - fa), cost = miss + 99 * fa;
            if (gap < eerGap) { eerGap = gap; eerThreshold = threshold; }
            if (cost < dcf) { dcf = cost; dcfThreshold = threshold; }
        }
        var metrics = SpeakerVerification.Measure(scores);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"dev positive={positives} negative={negatives} EER={metrics.Eer:R} EERthreshold={eerThreshold:R} minDCF={dcf:R} minDCFthreshold={dcfThreshold:R}"));
        // No tuning on the remaining speakers: report the frozen operating point.
        var holdout = trials.Where(t => !dev.Contains(t.First.Split('-')[0]) && !dev.Contains(t.Second.Split('-')[0]))
            .Select(t => (t.Same, Value: EmbeddingMath.Cosine(Load(t.First), Load(t.Second)))).ToArray();
        double holdMiss = (double)holdout.Count(s => s.Same && s.Value < dcfThreshold) / holdout.Count(s => s.Same);
        double holdFa = (double)holdout.Count(s => !s.Same && s.Value >= dcfThreshold) / holdout.Count(s => !s.Same);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"holdout trials={holdout.Length} miss={holdMiss:R} falseAccept={holdFa:R} DCF={holdMiss + 99 * holdFa:R}"));
        foreach (var vector in vectors.Values) Array.Clear(vector);
    }
}
