using System.Globalization;
using System.Text;
using System.Text.Json;

internal static class InterimEvaluation
{
    internal static void Run(string truthPath, string oursDirectory, string referenceDirectory, string output)
    {
        using var truth = JsonDocument.Parse(File.ReadAllText(truthPath));
        long total = 0, ourErrors = 0, referenceErrors = 0;
        int files = 0, languageCount = 0, languageMatches = 0, referenceWords = 0, matchedWords = 0;
        List<double> starts = [], ends = [];
        using var ourHypotheses = new StreamWriter(output + ".ours.jsonl", false, new UTF8Encoding(false));
        using var referenceHypotheses = new StreamWriter(output + ".reference.jsonl", false, new UTF8Encoding(false));
        foreach (var entry in truth.RootElement.EnumerateObject())
        {
            string oursPath = Path.Combine(oursDirectory, entry.Name + ".json"), refPath = Path.Combine(referenceDirectory, entry.Name + ".json");
            if (!File.Exists(oursPath) || !File.Exists(refPath)) continue;
            using var ours = JsonDocument.Parse(File.ReadAllText(oursPath));
            using var reference = JsonDocument.Parse(File.ReadAllText(refPath));
            var ourTranscript = ours.RootElement.GetProperty("transcript");
            string ourText = ourTranscript.GetProperty("text").GetString()!, referenceText = reference.RootElement.GetProperty("text").GetString()!;
            WriteHypothesis(ourHypotheses, entry.Name, ourText); WriteHypothesis(referenceHypotheses, entry.Name, referenceText);
            string[] target = Normalize(entry.Value.GetString()!);
            total += target.Length; ourErrors += Distance(target, Normalize(ourText)); referenceErrors += Distance(target, Normalize(referenceText)); files++;
            string ourLanguage = Path.Combine(oursDirectory, entry.Name + ".language"), referenceLanguage = Path.Combine(referenceDirectory, entry.Name + ".language");
            if (File.Exists(ourLanguage) && File.Exists(referenceLanguage)) { languageCount++; if (File.ReadAllText(ourLanguage) == File.ReadAllText(referenceLanguage)) languageMatches++; }
            var ourWords = ourTranscript.GetProperty("words").EnumerateArray().Where(w => w.GetProperty("start").ValueKind != JsonValueKind.Null).Select(w => new TimedWord(Key(w.GetProperty("text").GetString()!), w.GetProperty("start").GetDouble(), w.GetProperty("end").GetDouble())).Where(w => w.Text.Length > 0).ToArray();
            var refWords = reference.RootElement.GetProperty("segments").EnumerateArray().Where(s => s.TryGetProperty("words", out _)).SelectMany(s => s.GetProperty("words").EnumerateArray()).Select(w => new TimedWord(Key(w.GetProperty("word").GetString()!), w.GetProperty("start").GetDouble(), w.GetProperty("end").GetDouble())).Where(w => w.Text.Length > 0).ToArray();
            referenceWords += refWords.Length;
            foreach (var (a, b) in Match(ourWords, refWords)) { matchedWords++; starts.Add(Math.Abs(a.Start - b.Start) * 1000); ends.Add(Math.Abs(a.End - b.End) * 1000); }
        }
        using var report = new StreamWriter(output + ".metrics.txt", false);
        report.WriteLine("INTERIM normalizer=lowercase-unicode-alphanumeric; no paired CI (canonical evaluator pending).");
        report.WriteLine(string.Create(CultureInfo.InvariantCulture, $"files={files} reference_words={total} ours_errors={ourErrors} reference_errors={referenceErrors} ours_WER={100.0 * ourErrors / total:G9} reference_WER={100.0 * referenceErrors / total:G9}"));
        report.WriteLine($"English-only language agreement={languageMatches}/{languageCount}; no multilingual claim.");
        if (starts.Count > 0)
        {
            var all = starts.Concat(ends).ToArray();
            report.WriteLine(string.Create(CultureInfo.InvariantCulture, $"timing_matched_words={matchedWords}/{referenceWords} start_mean_ms={starts.Average():G9} start_median_ms={Median(starts):G9} end_mean_ms={ends.Average():G9} end_median_ms={Median(ends):G9} boundaries_within50_percent={100.0 * all.Count(v => v <= 50) / all.Length:G9} boundaries_within100_percent={100.0 * all.Count(v => v <= 100) / all.Length:G9}"));
        }
    }
    private static void WriteHypothesis(StreamWriter file, string id, string text)
    {
        using var memory = new MemoryStream();
        using (var json = new Utf8JsonWriter(memory)) { json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", text); json.WriteEndObject(); }
        file.WriteLine(Encoding.UTF8.GetString(memory.GetBuffer(), 0, (int)memory.Length));
    }
    private static string[] Normalize(string value)
    {
        var text = new StringBuilder();
        foreach (Rune rune in value.EnumerateRunes()) text.Append(Rune.IsLetterOrDigit(rune) ? Rune.ToLowerInvariant(rune).ToString() : " ");
        return text.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
    private static string Key(string value) => string.Concat(Normalize(value));
    private static int Distance(string[] target, string[] hypothesis)
    {
        int[] previous = Enumerable.Range(0, hypothesis.Length + 1).ToArray(), current = new int[hypothesis.Length + 1];
        for (int i = 1; i <= target.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= hypothesis.Length; j++) current[j] = Math.Min(previous[j - 1] + (target[i - 1] == hypothesis[j - 1] ? 0 : 1), Math.Min(previous[j] + 1, current[j - 1] + 1));
            (previous, current) = (current, previous);
        }
        return previous[^1];
    }
    private sealed record TimedWord(string Text, double Start, double End);
    // Exact lexical matches on a maximum-length monotone subsequence; unmatched words are reported, not assigned guessed boundaries.
    private static IEnumerable<(TimedWord, TimedWord)> Match(TimedWord[] ours, TimedWord[] reference)
    {
        int[,] lengths = new int[ours.Length + 1, reference.Length + 1];
        for (int i = ours.Length - 1; i >= 0; i--) for (int j = reference.Length - 1; j >= 0; j--) lengths[i, j] = ours[i].Text == reference[j].Text ? lengths[i + 1, j + 1] + 1 : Math.Max(lengths[i + 1, j], lengths[i, j + 1]);
        int a = 0, b = 0;
        while (a < ours.Length && b < reference.Length)
        {
            if (ours[a].Text == reference[b].Text) { yield return (ours[a++], reference[b++]); }
            else if (lengths[a + 1, b] >= lengths[a, b + 1]) a++; else b++;
        }
    }
    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray();
        return sorted.Length % 2 == 0 ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2 : sorted[sorted.Length / 2];
    }
}
