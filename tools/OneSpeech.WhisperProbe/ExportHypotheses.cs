using System.Text;
using System.Text.Json;

internal static class ExportHypotheses
{
    internal static void Run(string truthPath, string oursDirectory, string referenceDirectory, string output)
    {
        using var truth = JsonDocument.Parse(File.ReadAllText(truthPath));
        using var targets = new StreamWriter(output + ".truth.jsonl", false, new UTF8Encoding(false));
        using var ours = new StreamWriter(output + ".ours.jsonl", false, new UTF8Encoding(false));
        using var reference = new StreamWriter(output + ".reference.jsonl", false, new UTF8Encoding(false));
        IEnumerable<(string Name, string Text)> records = truth.RootElement.TryGetProperty("id", out var id)
            ? [(id.GetString()!, truth.RootElement.GetProperty("text").GetString()!)]
            : truth.RootElement.EnumerateObject().Select(p => (p.Name, p.Value.GetString()!));
        foreach (var entry in records)
        {
            using var native = JsonDocument.Parse(File.ReadAllText(Path.Combine(oursDirectory, entry.Name + ".json")));
            using var oracle = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDirectory, entry.Name + ".json")));
            Write(targets, entry.Name, entry.Text, []);
            var transcript = native.RootElement.GetProperty("transcript");
            var nativeWords = transcript.TryGetProperty("words", out var words) ? words.EnumerateArray().Select(w => new Word(w.GetProperty("text").GetString()!, Number(w.GetProperty("start")), Number(w.GetProperty("end")))).ToArray() : [];
            var referenceWords = oracle.RootElement.GetProperty("segments").EnumerateArray().Where(s => s.TryGetProperty("words", out _)).SelectMany(s => s.GetProperty("words").EnumerateArray()).Select(w => new Word(w.GetProperty("word").GetString()!, Number(w.GetProperty("start")), Number(w.GetProperty("end")))).ToArray();
            Write(ours, entry.Name, transcript.GetProperty("text").GetString()!, nativeWords);
            Write(reference, entry.Name, oracle.RootElement.GetProperty("text").GetString()!, referenceWords);
        }
    }
    private sealed record Word(string Text, double? Start, double? End);
    private static double? Number(JsonElement value) => value.ValueKind == JsonValueKind.Null ? null : value.GetDouble();
    private static void Write(StreamWriter file, string id, string text, Word[] words)
    {
        using var memory = new MemoryStream();
        using (var json = new Utf8JsonWriter(memory))
        {
            json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", text); json.WriteStartArray("words");
            foreach (var word in words)
            {
                json.WriteStartObject(); json.WriteString("text", word.Text);
                if (word.Start.HasValue) { json.WriteNumber("start", word.Start.Value); json.WriteNumber("end", word.End!.Value); }
                else { json.WriteNull("start"); json.WriteNull("end"); }
                json.WriteEndObject();
            }
            json.WriteEndArray(); json.WriteEndObject();
        }
        file.WriteLine(Encoding.UTF8.GetString(memory.GetBuffer(), 0, (int)memory.Length));
    }
}
