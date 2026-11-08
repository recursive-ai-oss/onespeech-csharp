using System.Text.Json;
using OneSpeech.Evaluation;

// Evaluation adapter: reuses the canonical normalizer and alignment, collecting raw
// boundary deltas across files rather than taking a median of per-file medians.
internal static class WhisperTimingSummary
{
    internal static void Write(string referencePath, string hypothesisPath, string output)
    {
        var reference = Hypotheses.Read(referencePath); var hypothesis = Hypotheses.Read(hypothesisPath);
        List<double> starts = [], ends = [];
        List<(string Id, string Text, double Start, double End)> outliers = [];
        int matched = 0;
        foreach (var entry in reference.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var a = entry.Value.Words ?? []; var b = hypothesis[entry.Key].Words ?? [];
            var alignment = Metrics.Align(a.Select(w => Normalizer.Normalize(w.Text, "en")).ToArray(), b.Select(w => Normalizer.Normalize(w.Text, "en")).ToArray());
            foreach (var pair in alignment.Matches.Where(m => m.Equal && Normalizer.Normalize(a[m.Reference].Text, "en").Length > 0))
            {
                matched++; var expected = a[pair.Reference]; var actual = b[pair.Hypothesis];
                if (expected.Start is not { } rs || expected.End is not { } re || actual.Start is not { } hs || actual.End is not { } he) continue;
                double start = Math.Abs(rs - hs) * 1000, end = Math.Abs(re - he) * 1000;
                starts.Add(start); ends.Add(end); outliers.Add((entry.Key, expected.Text, start, end));
            }
        }
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("normalizer", "en/1"); json.WriteNumber("files", reference.Count); json.WriteNumber("matched", matched); json.WriteNumber("timed", starts.Count);
        void Summarize(string name, List<double> values)
        {
            var sorted = values.Order().ToArray(); json.WriteStartObject(name);
            if (sorted.Length > 0) { json.WriteNumber("meanMs", values.Average()); json.WriteNumber("medianMs", (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2); json.WriteNumber("within50", (double)values.Count(v => v <= 50 + 1e-9) / values.Count); json.WriteNumber("within100", (double)values.Count(v => v <= 100 + 1e-9) / values.Count); }
            json.WriteEndObject();
        }
        Summarize("start", starts); Summarize("end", ends);
        json.WriteStartArray("largestErrors");
        foreach (var item in outliers.OrderByDescending(o => Math.Max(o.Start, o.End)).Take(20))
        {
            json.WriteStartObject(); json.WriteString("id", item.Id); json.WriteString("word", item.Text); json.WriteNumber("startErrorMs", item.Start); json.WriteNumber("endErrorMs", item.End); json.WriteEndObject();
        }
        json.WriteEndArray(); json.WriteEndObject();
    }
}
