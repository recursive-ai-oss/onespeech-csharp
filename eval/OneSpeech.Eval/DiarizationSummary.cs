using System.Text.Json;

namespace OneSpeech.Evaluation;

internal static class DiarizationSummary
{
    internal static void Run(string root)
    {
        var rows = new List<Row>();
        foreach (string path in Directory.EnumerateFiles(Path.Combine(root, "holdout"), "scores.jsonl", SearchOption.AllDirectories))
            foreach (string line in File.ReadLines(path))
            {
                using var json = JsonDocument.Parse(line); var e = json.RootElement;
                double D(string name) => e.GetProperty(name).GetDouble();
                string id = e.GetProperty("id").GetString()!;
                rows.Add(new Row(id.StartsWith("ES", StringComparison.Ordinal) ? "AMI-test" : "AliMeeting-eval", id, e.GetProperty("profile").GetString()!, D("referenceSeconds"), D("missSeconds"), D("falseAlarmSeconds"), D("confusionSeconds"), D("jaccardErrorSum"), e.GetProperty("referenceSpeakers").GetInt32()));
            }
        using var file = File.Create(Path.Combine(root, "floor-summary.json")); using var w = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        w.WriteStartObject(); w.WriteString("status", "first-measurement floors; no reference parity or absolute quality pass claimed"); w.WriteString("bootstrap", "1000 resamples over meetings, seed1729, floor(.025*(B-1))/ceil(.975*(B-1)) percentile endpoints; descriptive conditional intervals with only two meetings per dataset"); w.WriteStartArray("datasets");
        foreach (var group in rows.GroupBy(r => (r.Dataset, r.Profile)).OrderBy(g => g.Key.Dataset, StringComparer.Ordinal).ThenBy(g => g.Key.Profile, StringComparer.Ordinal))
        {
            var data = group.OrderBy(r => r.Id, StringComparer.Ordinal).ToArray();
            double Reference(IEnumerable<Row> sample) => sample.Sum(r => r.Reference);
            double Der(IEnumerable<Row> sample) => sample.Sum(r => r.Miss + r.Fa + r.Confusion) / Reference(sample);
            double Jer(IEnumerable<Row> sample) => sample.Sum(r => r.JerSum) / sample.Sum(r => r.Speakers);
            w.WriteStartObject(); w.WriteString("dataset", group.Key.Dataset); w.WriteString("profile", group.Key.Profile); w.WriteNumber("meetings", data.Length);
            w.WriteNumber("referenceSeconds", Reference(data)); w.WriteNumber("missSeconds", data.Sum(r => r.Miss)); w.WriteNumber("falseAlarmSeconds", data.Sum(r => r.Fa)); w.WriteNumber("confusionSeconds", data.Sum(r => r.Confusion)); w.WriteNumber("der", Der(data)); w.WriteNumber("jer", Jer(data));
            if (data.Length < 2) { w.WriteNull("derCI95"); w.WriteNull("jerCI95"); }
            else
            {
                var der = new double[1000]; var jer = new double[1000]; var random = new Random(1729);
                for (int b = 0; b < der.Length; b++) { var sample = Enumerable.Range(0, data.Length).Select(_ => data[random.Next(data.Length)]).ToArray(); der[b] = Der(sample); jer[b] = Jer(sample); }
                Array.Sort(der); Array.Sort(jer);
                void CI(string name, double[] values) { w.WriteStartArray(name); w.WriteNumberValue(values[(int)Math.Floor(0.025 * (values.Length - 1))]); w.WriteNumberValue(values[(int)Math.Ceiling(0.975 * (values.Length - 1))]); w.WriteEndArray(); }
                CI("derCI95", der); CI("jerCI95", jer);
            }
            w.WriteEndObject(); Console.WriteLine(FormattableString.Invariant($"{group.Key.Dataset} {group.Key.Profile}: DER={100 * Der(data):F4}% JER={100 * Jer(data):F4}% ({data.Length} meetings)"));
        }
        w.WriteEndArray(); w.WriteEndObject();
    }
    private sealed record Row(string Dataset, string Id, string Profile, double Reference, double Miss, double Fa, double Confusion, double JerSum, int Speakers);
}
