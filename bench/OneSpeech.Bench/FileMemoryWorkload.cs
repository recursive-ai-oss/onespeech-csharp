using System.Diagnostics;
using System.Text.Json;
using OneSpeech;

// Unlike a warm structural-ratio row, this probes the actual file helper, whose
// recognizer and timing sessions are released between phases. Loading is included.
internal static class FileMemoryWorkload
{
    internal static void Run(string file, string model, int threads, string output)
    {
        if (threads is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(threads));
        Directory.CreateDirectory(output);
        string beforeLoad = Machine.Load();
        GraphTiming.ThreadCount = threads;
        GraphTiming.Reset();
        string Phase()
        {
            var phases = GraphTiming.PhaseSnapshot();
            return phases[3] != 0 ? "complete" : phases[2] != 0 ? "diarization" : phases[1] != 0 ? "alignment" : phases[0] != 0 ? "recognition" : "loading";
        }
        var samples = new MemorySamples(Path.Combine(output, "file.memory.csv"), Phase);
        var clock = Stopwatch.StartNew();
        Transcript transcript;
        try { transcript = Speech.Transcribe(file, language: "en", model: model); }
        finally { clock.Stop(); samples.Dispose(); }
        // Keep the growing result alive throughout measurement, rather than hiding
        // output memory from the probe. No audio, embeddings or transcript is saved.
        using var stream = File.Create(Path.Combine(output, "file.json"));
        using var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteString("model", model); json.WriteNumber("threads", threads);
        json.WriteNumber("elapsedIncludingLoadingSeconds", clock.Elapsed.TotalSeconds);
        json.WriteNumber("words", transcript.Words.Length);
        json.WriteNumber("sampledPeakWorkingSetBytes", samples.Peak);
        samples.WriteSlopes(json);
        Machine.Write(json, beforeLoad, samples.Loads);
        json.WriteEndObject();
        Console.WriteLine($"File memory probe: {transcript.Words.Length} words, {clock.Elapsed.TotalSeconds:F3} s including loading.");
        GC.KeepAlive(transcript);
    }
}
