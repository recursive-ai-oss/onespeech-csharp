using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Evaluation;

internal static class NemotronBatch
{
    internal static void ScoreOnly(string truthPath, string hypothesisPath, string referencePath)
    {
        using var truthFile = File.OpenRead(truthPath); using var hypothesisFile = File.OpenRead(hypothesisPath);
        var truth = Rttm.Read(truthFile); var hypothesis = Rttm.Read(hypothesisFile);
        using var referenceFile = File.OpenRead(referencePath); using var doc = JsonDocument.Parse(referenceFile);
        string id = truth[0].Recording;
        var reference = doc.RootElement.GetProperty("segments").EnumerateArray().Select(s => new DiarizationSegment(id, "1", "speaker_" + s.GetProperty("Speaker").GetInt32(), s.GetProperty("Start").GetDouble(), s.GetProperty("End").GetDouble())).ToArray();
        var ours = DiarizationMetrics.Score(truth, hypothesis); var oracle = DiarizationMetrics.Score(truth, reference);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{id} DER={100 * ours.Der:F4}/{100 * oracle.Der:F4} JER={100 * ours.Jer:F4}/{100 * oracle.Jer:F4}"));
    }
    internal static void Run(string directory, string audio, string output, string truthPath, string referencePath)
    {
        Directory.CreateDirectory(output);
        var entry = ModelCatalog.Default.Get("nvidia/Nemotron-3-Diarization");
        var info = new ModelInfo { Id = entry.Id, Revision = entry.Revision, Source = entry.Source, Tier = entry.Tier, License = entry.License, Attribution = entry.Attribution };
        using var model = NemotronDiarizer.Load(new ResolvedModel(directory, info), threads: 4);
        string id = Path.GetFileNameWithoutExtension(audio);
        using var memory = new StreamWriter(Path.Combine(output, id + ".memory.tsv"));
        memory.WriteLine("audio_read_seconds\tworking_set_bytes\tmanaged_bytes\tload_average");
        using var process = Process.GetCurrentProcess();
        using var source = AudioFileSource.Open(audio, CancellationToken.None);
        long samples = 0, nextSample = 0;
        IEnumerable<AudioBuffer> Blocks()
        {
            foreach (var block in VadChunker.Blocks(source, CancellationToken.None))
            {
                samples += block.FrameCount;
                if (samples >= nextSample)
                {
                    process.Refresh();
                    string load = File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : "unavailable";
                    memory.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{samples / 16000d:R}\t{process.WorkingSet64}\t{GC.GetTotalMemory(false)}\t{load}")); memory.Flush();
                    nextSample += 30 * 16000;
                }
                yield return block;
            }
        }
        var turns = model.DiarizeBlocks(Blocks(), CancellationToken.None);
        var hypothesis = turns.Select(t => new DiarizationSegment(id, "1", t.Speaker.Id, t.Start.TotalSeconds, t.End.TotalSeconds)).ToArray();
        using (var rttm = new StreamWriter(Path.Combine(output, id + ".rttm")))
            foreach (var turn in hypothesis)
                rttm.WriteLine(string.Create(CultureInfo.InvariantCulture, $"SPEAKER {id} 1 {turn.Start:F6} {turn.End - turn.Start:F6} <NA> <NA> {turn.Speaker} <NA> <NA>"));
        using var input = File.OpenRead(truthPath);
        var truth = Rttm.Read(input);
        using var referenceFile = File.OpenRead(referencePath);
        using var referenceDocument = JsonDocument.Parse(referenceFile);
        var reference = referenceDocument.RootElement.GetProperty("segments").EnumerateArray().Select(s => new DiarizationSegment(id, "1", "speaker_" + s.GetProperty("Speaker").GetInt32(), s.GetProperty("Start").GetDouble(), s.GetProperty("End").GetDouble())).ToArray();
        var ours = DiarizationMetrics.Score(truth, hypothesis);
        var oracle = DiarizationMetrics.Score(truth, reference);
        var difference = DiarizationMetrics.Score(reference, hypothesis);
        using var metrics = File.Create(Path.Combine(output, id + ".metrics.json"));
        using var writer = new Utf8JsonWriter(metrics, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject(); writer.WriteString("id", id); writer.WriteString("model", entry.Id); writer.WriteString("revision", entry.Revision);
        writer.WriteString("precision", "fp32"); writer.WriteString("oracle", "Transformers 5.22.2"); writer.WriteNumber("threads", 4);
        writer.WriteBoolean("overlapScored", true); writer.WriteNumber("collarSeconds", 0); writer.WriteNumber("audioSeconds", samples / 16000d);
        writer.WriteNumber("ourDer", ours.Der ?? 0); writer.WriteNumber("ourJer", ours.Jer ?? 0); writer.WriteNumber("referenceDer", oracle.Der ?? 0); writer.WriteNumber("referenceJer", oracle.Jer ?? 0);
        writer.WriteNumber("referenceHypothesisDer", difference.Der ?? 0); writer.WriteNumber("referenceHypothesisJer", difference.Jer ?? 0);
        writer.WriteBoolean("derWithinHalfPoint", Math.Abs((ours.Der ?? 0) - (oracle.Der ?? 0)) <= .005);
        writer.WriteBoolean("jerWithinHalfPoint", Math.Abs((ours.Jer ?? 0) - (oracle.Jer ?? 0)) <= .005);
        writer.WriteNumber("turns", turns.Length); writer.WriteEndObject(); writer.Flush();
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{id} DER={100 * ours.Der:F4}/{100 * oracle.Der:F4} JER={100 * ours.Jer:F4}/{100 * oracle.Jer:F4} vs-reference-DER={100 * difference.Der:F4} turns={turns.Length}"));
    }
}
