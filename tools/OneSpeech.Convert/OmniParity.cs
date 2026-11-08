using OneSpeech.Evaluation;
using System.Text.Json;

internal static class OmniParity
{
    internal static void Score(string directory, string nativeFile, string referenceFile, string destination)
    {
        var truth = Hypotheses.LibriSpeech(directory).ToDictionary(u => u.Id, StringComparer.Ordinal);
        var native = Hypotheses.Read(nativeFile);
        var reference = Hypotheses.Read(referenceFile);
        string[] ids = truth.Keys.Order(StringComparer.Ordinal).Take(200).ToArray();
        if (native.Count != 200 || reference.Count != 200 || ids.Any(id => !native.ContainsKey(id) || !reference.ContainsKey(id)))
            throw new InvalidDataException("Expected exactly the fixed first 200 IDs on both sides.");
        ErrorCounts[] ours = ids.Select(id => Metrics.Score(Normalizer.Normalize(truth[id].Text, "en"), Normalizer.Normalize(native[id].Text, "en"), true)).ToArray();
        ErrorCounts[] theirs = ids.Select(id => Metrics.Score(Normalizer.Normalize(truth[id].Text, "en"), Normalizer.Normalize(reference[id].Text, "en"), true)).ToArray();
        var ci = Metrics.Bootstrap(ours, theirs, 1000, 1729);
        ErrorCounts Sum(ErrorCounts[] rows) => rows.Aggregate(new ErrorCounts(0, 0, 0, 0), (a, b) => a + b);
        using var output = File.Create(destination);
        using var writer = new Utf8JsonWriter(output, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject(); writer.WriteString("normalizer", "en/1"); writer.WriteString("metric", "CER");
        writer.WriteNumber("utterances", ids.Length); writer.WriteNumber("probe20Mismatches", ids.Take(20).Count(id => native[id].Text != reference[id].Text));
        writer.WriteNumber("all200TranscriptMismatches", ids.Count(id => native[id].Text != reference[id].Text));
        writer.WriteNumber("ours", Sum(ours).Rate!.Value); writer.WriteNumber("reference", Sum(theirs).Rate!.Value);
        writer.WriteNumber("resamples", 1000); writer.WriteNumber("seed", 1729);
        if (ci.Difference is { } interval) { writer.WriteNumber("deltaLower", interval.Lower); writer.WriteNumber("deltaUpper", interval.Upper); }
        writer.WriteString("verdict", ci.Verdict); writer.WriteEndObject();
    }
}
