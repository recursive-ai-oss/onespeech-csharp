using System.Text.Json;
using OneSpeech.Audio;
using OneSpeech.Diarization;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

// Frozen file-helper policy; unlike the development grid this never fits parameters.
internal static class DiarizationGate
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    internal static void Default(string graph, string audio, string output)
    {
        if (File.Exists(output)) return;
        using var embedder = DiarizationRun.LoadEmbedder(graph, 4);
        using var vad = new SileroVad(SileroVadOptions.Diarization);
        using var model = new ClusteringDiarizer(embedder, vad);
        using var source = AudioFileSource.Open(audio, CancellationToken.None);
        var turns = model.DiarizeBlocks(VadChunker.Blocks(source, CancellationToken.None), CancellationToken.None);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        DiarizationRun.WriteTurns(output + ".partial", Path.GetFileNameWithoutExtension(audio), turns);
        File.Move(output + ".partial", output);
    }

    internal static void Score(string root, string ours, string reference, string output)
    {
        var rows = File.ReadLines(Path.Combine(root, "manifest.jsonl")).Select(line =>
        {
            using var json = JsonDocument.Parse(line);
            string id = json.RootElement.GetProperty("id").GetString()!;
            var entry = DiarizationRun.ReadEntry(root, id); DiarizationRun.VerifyEntry(root, entry);
            using var truthFile = File.OpenRead(Path.Combine(root, entry.Reference)); var truth = Rttm.Read(truthFile);
            using var oursFile = File.OpenRead(Path.Combine(ours, id + ".rttm"));
            using var referenceFile = File.OpenRead(Path.Combine(reference, id + ".rttm"));
            return new Row(id, DiarizationMetrics.Score(truth, Rttm.Read(oursFile)), DiarizationMetrics.Score(truth, Rttm.Read(referenceFile)));
        }).OrderBy(r => r.Id, StringComparer.Ordinal).ToArray();
        if (rows.Length == 0) throw new InvalidDataException("No meetings.");
        double Rate(IEnumerable<DiarizationScore> scores, bool jer)
        {
            var data = scores.ToArray();
            return jer ? data.Sum(s => s.JaccardErrorSum) / data.Sum(s => s.ReferenceSpeakers)
                : data.Sum(s => s.MissSeconds + s.FalseAlarmSeconds + s.ConfusionSeconds) / data.Sum(s => s.ReferenceSeconds);
        }
        object Metric(bool jer)
        {
            double a = Rate(rows.Select(r => r.Ours), jer), b = Rate(rows.Select(r => r.Reference), jer);
            var random = new Random(1729); var delta = new double[1000];
            for (int i = 0; i < delta.Length; i++)
            {
                var sample = Enumerable.Range(0, rows.Length).Select(_ => rows[random.Next(rows.Length)]).ToArray();
                delta[i] = Rate(sample.Select(r => r.Ours), jer) - Rate(sample.Select(r => r.Reference), jer);
            }
            Array.Sort(delta); double low = delta[24], high = delta[975];
            return new { ours = a, reference = b, difference = a - b, ci95 = new[] { low, high }, pass = rows.Length > 1 ? low <= 0 && high >= 0 : Math.Abs(a - b) <= .005 };
        }
        using var stream = File.Create(output);
        JsonSerializer.Serialize(stream, new { meetings = rows.Length, seed = 1729, resamples = 1000, collarSeconds = 0, overlapScored = true, der = Metric(false), jer = Metric(true), rows }, JsonOptions);
    }
    private sealed record Row(string Id, DiarizationScore Ours, DiarizationScore Reference);
}
