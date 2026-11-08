using System.Text.Json;
using OneSpeech.Evaluation;

internal static class ParakeetEvaluation
{
    internal static void Prepare(string corpus, string ours, string reference, string output)
    {
        Directory.CreateDirectory(output);
        var truth = Hypotheses.LibriSpeech(corpus).ToDictionary(row => row.Id, StringComparer.Ordinal);
        List<Utterance> expected = [], actual = [], oracle = [];
        int tokenMatches = 0, frameMatches = 0, tokenCount = 0;
        var paths = Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(200).ToArray();
        foreach (string path in paths)
        {
            string id = Path.GetFileNameWithoutExtension(path);
            using var oursDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(ours, id + ".json")));
            using var referenceDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, id + ".json")));
            var a = oursDocument.RootElement; var b = referenceDocument.RootElement;
            actual.Add(new(id, a.GetProperty("text").GetString()!, a.GetProperty("words").EnumerateArray().Select(word => new EvalWord(word.GetProperty("text").GetString()!, word.GetProperty("start").GetDouble(), word.GetProperty("end").GetDouble())).ToArray()));
            oracle.Add(new(id, b.GetProperty("text").GetString()!, b.GetProperty("timestamp").GetProperty("word").EnumerateArray().Select(word => new EvalWord(word.GetProperty("word").GetString()!, word.GetProperty("start").GetDouble(), word.GetProperty("end").GetDouble())).ToArray()));
            expected.Add(truth[id]);
            var oursTokens = a.GetProperty("tokens").EnumerateArray().Select(item => item.GetInt32()).ToArray();
            var referenceTokens = b.GetProperty("tokens").EnumerateArray().Select(item => item.GetInt32()).ToArray();
            bool equal = oursTokens.SequenceEqual(referenceTokens); if (equal) tokenMatches++; else Console.WriteLine("Token divergence: " + id);
            tokenCount += referenceTokens.Length;
            var oursFrames = a.GetProperty("frames").EnumerateArray().Select(item => item.GetInt32());
            var referenceFrames = b.GetProperty("timestamp").GetProperty("timestep").EnumerateArray().Select(item => item.GetInt32());
            if (equal && oursFrames.SequenceEqual(referenceFrames)) frameMatches++;
        }
        Hypotheses.Write(Path.Combine(output, "truth.jsonl"), expected);
        Hypotheses.Write(Path.Combine(output, "ours.jsonl"), actual);
        Hypotheses.Write(Path.Combine(output, "nemo.jsonl"), oracle);
        Console.WriteLine($"Token-identical utterances {tokenMatches}/{paths.Length}; frame-identical {frameMatches}/{paths.Length}; reference tokens {tokenCount}");
        Timing(Path.Combine(output, "nemo.jsonl"), Path.Combine(output, "ours.jsonl"));
    }

    internal static void Timing(string reference, string hypothesis)
    {
        var truth = Hypotheses.Read(reference); var ours = Hypotheses.Read(hypothesis);
        List<BoundarySummary> rows = [];
        foreach (var pair in ours)
        {
            if (!truth.TryGetValue(pair.Key, out var expected)) throw new InvalidDataException("Missing timing reference " + pair.Key);
            rows.Add(TimingMetrics.Score(expected.Words!, pair.Value.Words!, "en"));
        }
        int count = rows.Sum(row => row.Timed);
        double Average(Func<BoundarySummary, double?> selector) => rows.Sum(row => (selector(row) ?? 0) * row.Timed) / count;
        Console.WriteLine($"Timing words matched={rows.Sum(row => row.Matched)} timed={count} missing={rows.Sum(row => row.Missing)}; mean start/end ms={Average(row => row.MeanStartMs):F6}/{Average(row => row.MeanEndMs):F6}; within50 start/end={100 * Average(row => row.StartWithin50):F6}/{100 * Average(row => row.EndWithin50):F6}%; within100 start/end={100 * Average(row => row.StartWithin100):F6}/{100 * Average(row => row.EndWithin100):F6}%");
    }
}
