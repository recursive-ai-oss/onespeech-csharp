using System.Text.Json;
using OneSpeech.Evaluation;

internal static class MoonshineLatency
{
    internal static void Run(string hypotheses, string timings, string output)
    {
        var values = new List<double>();
        var recordings = new List<(double Sum, int Count)>();
        int referenceCount = 0, hypothesisCount = 0;
        foreach (string path in Directory.EnumerateFiles(hypotheses, "*.json").Order(StringComparer.Ordinal))
        {
            string truth = Path.Combine(timings, Path.GetFileName(path));
            if (!File.Exists(truth)) throw new FileNotFoundException("Missing estimated word-end reference.", truth);
            using var native = JsonDocument.Parse(File.ReadAllBytes(path));
            using var timing = JsonDocument.Parse(File.ReadAllBytes(truth));
            var reference = timing.RootElement.GetProperty("words").EnumerateArray().SelectMany(w => Tokens(w.GetProperty("text").GetString()!).Select(t => (Text: t, Time: w.GetProperty("end").GetDouble()))).ToArray();
            var hypothesis = native.RootElement.GetProperty("finals").EnumerateArray().SelectMany(f => Tokens(f.GetProperty("text").GetString()!).Select(t => (Text: t, Time: f.GetProperty("available").GetDouble()))).ToArray();
            referenceCount += reference.Length; hypothesisCount += hypothesis.Length;
            int[,] cost = new int[reference.Length + 1, hypothesis.Length + 1];
            for (int i = 0; i <= reference.Length; i++) cost[i, 0] = i;
            for (int j = 0; j <= hypothesis.Length; j++) cost[0, j] = j;
            for (int i = 1; i <= reference.Length; i++) for (int j = 1; j <= hypothesis.Length; j++) cost[i, j] = Math.Min(cost[i - 1, j - 1] + (reference[i - 1].Text == hypothesis[j - 1].Text ? 0 : 1), Math.Min(cost[i - 1, j] + 1, cost[i, j - 1] + 1));
            int r = reference.Length, h = hypothesis.Length, start = values.Count;
            while (r > 0 || h > 0)
            {
                bool same = r > 0 && h > 0 && reference[r - 1].Text == hypothesis[h - 1].Text;
                if (r > 0 && h > 0 && cost[r, h] == cost[r - 1, h - 1] + (same ? 0 : 1)) { if (same) values.Add((hypothesis[h - 1].Time - reference[r - 1].Time) * 1000); r--; h--; }
                else if (r > 0 && cost[r, h] == cost[r - 1, h] + 1) r--;
                else h--;
            }
            recordings.Add((values.Skip(start).Sum(), values.Count - start));
        }
        values.Sort();
        var random = new Random(1729);
        double[] bootstrap = new double[1000];
        for (int sample = 0; sample < bootstrap.Length; sample++)
        {
            double sum = 0; int count = 0;
            for (int i = 0; i < recordings.Count; i++) { var row = recordings[random.Next(recordings.Count)]; sum += row.Sum; count += row.Count; }
            bootstrap[sample] = count == 0 ? double.NaN : sum / count;
        }
        Array.Sort(bootstrap);
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("timingReference", "openai-whisper20250625 tiny fp32 DTW estimates; not ground truth"); json.WriteString("normalizer", "en/1"); json.WriteNumber("recordings", recordings.Count); json.WriteNumber("referenceTokens", referenceCount); json.WriteNumber("hypothesisTokens", hypothesisCount); json.WriteNumber("matchedTokens", values.Count); json.WriteNumber("negativeLatencies", values.Count(v => v < 0));
        if (values.Count > 0) { json.WriteNumber("meanMs", values.Average()); json.WriteNumber("p50Ms", values[(values.Count - 1) / 2]); json.WriteNumber("p95Ms", values[(int)Math.Ceiling(values.Count * .95) - 1]); if (double.IsFinite(bootstrap[25])) { json.WriteNumber("meanCiLowMs", bootstrap[25]); json.WriteNumber("meanCiHighMs", bootstrap[974]); } }
        json.WriteNumber("bootstrapResamples", 1000); json.WriteNumber("bootstrapSeed", 1729); json.WriteEndObject();
        static string[] Tokens(string text) => Normalizer.Normalize(text, "en").Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}
