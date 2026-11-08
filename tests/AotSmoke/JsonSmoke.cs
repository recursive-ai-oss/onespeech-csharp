using System.Text.Json;
using System.Text.Json.Serialization;
using OneSpeech;

internal static class JsonSmoke
{
    internal static void Run(Transcript transcript)
    {
        string json = JsonSerializer.Serialize(transcript, SmokeJsonContext.Default.Transcript);
        if (json != transcript.ToJson() || JsonSerializer.Deserialize(json, SmokeJsonContext.Default.Transcript) != transcript)
            throw new InvalidOperationException("Canonical System.Text.Json round trip failed.");
        var minimum = Transcript.FromJson("""{"schemaVersion":1,"transcript":{"text":"x","segments":[{"text":"x"}]}}""");
        if (!minimum.Words.IsEmpty || !minimum.Segments[0].Words.IsEmpty) throw new InvalidOperationException("Missing-array defaults failed.");
        Console.WriteLine("Canonical System.Text.Json NativeAOT smoke passed.");
    }
}

[JsonSerializable(typeof(Transcript))]
internal sealed partial class SmokeJsonContext : JsonSerializerContext;
