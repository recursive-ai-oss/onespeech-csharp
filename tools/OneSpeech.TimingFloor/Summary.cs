using System.Text.Json;
using OneSpeech.Evaluation;

// Pool absolute boundary errors over words, not an average of per-file medians.
internal static class Summary
{
    internal static void Write(string split)
    {
        var truth = Hypotheses.Read(Path.Combine(split, "truth.jsonl"));
        string[] methods = ["ctc", "dtw", "oracle"];
        var hypotheses = methods.ToDictionary(m => m, m => Hypotheses.Read(Path.Combine(split, m + ".jsonl")), StringComparer.Ordinal);
        var pairs = methods.ToDictionary(m => m, _ => new Dictionary<(string Id, int Index), EvalWord>(), StringComparer.Ordinal);
        foreach (var (id, reference) in truth)
        {
            foreach (string method in methods)
            {
                var expected = reference.Words!; var actual = hypotheses[method][id].Words!;
                var alignment = Metrics.Align(expected.Select(w => Normalizer.Normalize(w.Text, "en")).ToArray(), actual.Select(w => Normalizer.Normalize(w.Text, "en")).ToArray());
                foreach (var match in alignment.Matches.Where(m => m.Equal && Normalizer.Normalize(expected[m.Reference].Text, "en").Length != 0))
                    pairs[method].Add((id, match.Reference), actual[match.Hypothesis]);
            }
        }
        var keys = truth.SelectMany(t => t.Value.Words!.Select((w, i) => (Key: (Id: t.Key, Index: i), Word: w))).Where(t => Normalizer.Normalize(t.Word.Text, "en").Length != 0).ToArray();
        var common = keys.Where(k => k.Word.Start is not null && methods.All(m => pairs[m].TryGetValue(k.Key, out var w) && w.Start is not null)).ToArray();
        using var output = File.Create(Path.Combine(split, "summary.json")); using var json = new Utf8JsonWriter(output, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("normalizer", "en/1"); json.WriteNumber("files", truth.Count); json.WriteNumber("writtenReferenceWords", truth.Values.Sum(v => v.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length)); json.WriteNumber("truthWords", keys.Length); json.WriteNumber("commonTimedWords", common.Length);
        foreach (string method in methods)
        {
            json.WriteStartObject(method);
            json.WriteNumber("hypothesisWords", hypotheses[method].Values.Sum(v => v.Words!.Length));
            json.WriteNumber("hypothesisUntimed", hypotheses[method].Values.Sum(v => v.Words!.Count(w => w.Start is null)));
            json.WriteNumber("unmatchedTruthWords", keys.Length - pairs[method].Count);
            var matched = keys.Where(k => pairs[method].ContainsKey(k.Key)).ToArray();
            WriteScore("allMatched", ScorePairs(matched.Select(k => (k.Word, pairs[method][k.Key]))));
            WriteScore("common", ScorePairs(common.Select(k => (k.Word, pairs[method][k.Key]))));
            json.WriteEndObject();
        }
        json.WriteEndObject();
        void WriteScore(string name, BoundarySummary value)
        {
            json.WriteStartObject(name); json.WriteNumber("matched", value.Matched); json.WriteNumber("timed", value.Timed); json.WriteNumber("missing", value.Missing);
            void Number(string key, double? x) { if (x is { } v) json.WriteNumber(key, v); else json.WriteNull(key); }
            Number("meanStartMs", value.MeanStartMs); Number("medianStartMs", value.MedianStartMs); Number("meanEndMs", value.MeanEndMs); Number("medianEndMs", value.MedianEndMs);
            Number("startWithin50", value.StartWithin50); Number("startWithin100", value.StartWithin100); Number("endWithin50", value.EndWithin50); Number("endWithin100", value.EndWithin100); json.WriteEndObject();
        }
    }

    private static BoundarySummary ScorePairs(IEnumerable<(EvalWord Reference, EvalWord Hypothesis)> pairs)
    {
        // The utterance alignments above have already fixed the word correspondence.
        // Never re-align a whole corpus: the canonical trellis is bounded to 16M cells.
        List<double> start = [], end = []; int matched = 0;
        foreach (var pair in pairs)
        {
            var score = TimingMetrics.Score([pair.Reference], [pair.Hypothesis], "en");
            matched += score.Matched;
            if (score.MeanStartMs is not { } s || score.MeanEndMs is not { } e) continue;
            start.Add(s); end.Add(e);
        }
        static double? Mean(List<double> values) => values.Count == 0 ? null : values.Average();
        static double? Median(List<double> values)
        {
            if (values.Count == 0) return null;
            var sorted = values.Order().ToArray(); return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
        }
        static double? Within(List<double> values, int ms) => values.Count == 0 ? null : (double)values.Count(v => v <= ms + 1e-9) / values.Count;
        return new(matched, start.Count, matched - start.Count, Mean(start), Median(start), Mean(end), Median(end), Within(start, 50), Within(start, 100), Within(end, 50), Within(end, 100));
    }
}
