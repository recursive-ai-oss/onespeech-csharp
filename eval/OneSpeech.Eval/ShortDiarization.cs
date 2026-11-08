using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using OneSpeech.Audio;
using OneSpeech.Diarization;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

// Deterministic, worker-only short recording preparation and scoring. No embeddings are persisted.
internal static class ShortDiarization
{
    private static readonly CancellationToken Token = CancellationToken.None;
    internal static int Run(string[] args)
    {
        if (args is ["prepare-ls", var corpus, var output]) { Prepare(corpus, output); return 0; }
        if (args is ["prepare-dev", var root, var target]) { PrepareDev(root, target); return 0; }
        if (args is not ["score", var graph, var manifest, var destination, var mode])
            throw new ArgumentException("short-diarization prepare-ls CORPUS OUTPUT | prepare-dev ROOT OUTPUT | score GRAPH MANIFEST OUTPUT MODE");
        using var embedder = DiarizationRun.LoadEmbedder(graph, threads: 1);
        using var vad = new SileroVad(mode == "low" ? null : SileroVadOptions.Diarization);
        var options = new ClusteringOptions { Resegment = false };
        if (mode.StartsWith("ahc-", StringComparison.Ordinal)) options = options with { Method = ClusteringMethod.Agglomerative, AgglomerativeThreshold = double.Parse(mode[4..], CultureInfo.InvariantCulture) };
        var diarizer = new ClusteringDiarizer(embedder, vad, options,
            mode == "low" ? null : new CosineResegmenter(0.4, 2));
        using var writer = new StreamWriter(destination);
        writer.WriteLine("id\texpected\tpredicted\tder\tmiss\tfalseAlarm\tconfusion\treferenceSeconds");
        foreach (string row in File.ReadLines(manifest))
        {
            string[] fields = row.Split('\t'); string id = fields[0];
            using var source = AudioFileSource.Open(fields[1], Token);
            var turns = mode == "transcribe"
                ? Speech.Transcribe(fields[1], Token, language: "en", model: "openai/whisper-tiny").Turns
                : mode.StartsWith("json:", StringComparison.Ordinal)
                ? Transcript.FromJson(File.ReadAllBytes(Path.Combine(mode[5..], id + ".json"))).Turns
                : diarizer.DiarizeBlocks(DiarizationRun.Blocks(source), Token);
            using var referenceFile = File.OpenRead(fields[2]); var reference = Rttm.Read(referenceFile);
            var predicted = turns.Select(t => new DiarizationSegment(id, "1", t.Speaker.Id, t.Start.TotalSeconds, t.End.TotalSeconds)).ToArray();
            var score = DiarizationMetrics.Score(reference, predicted);
            writer.WriteLine(FormattableString.Invariant($"{id}\t{reference.Select(s => s.Speaker).Distinct().Count()}\t{turns.Select(t => t.Speaker.Id).Distinct().Count()}\t{score.Der:R}\t{score.MissSeconds:R}\t{score.FalseAlarmSeconds:R}\t{score.ConfusionSeconds:R}\t{score.ReferenceSeconds:R}"));
            writer.Flush(); Console.WriteLine(id + " " + turns.Select(t => t.Speaker.Id).Distinct().Count());
        }
        return 0;
    }
    private static float[] Load(string path)
    {
        using var source = AudioFileSource.Open(path, Token);
        return DiarizationRun.Blocks(source).SelectMany(b => b.Samples.ToArray()).ToArray();
    }
    private static void Wave(string path, float[] samples)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write("RIFF"u8); w.Write(36 + samples.Length * 4); w.Write("WAVEfmt "u8); w.Write(16); w.Write((short)3); w.Write((short)1); w.Write(16000); w.Write(64000); w.Write((short)4); w.Write((short)32); w.Write("data"u8); w.Write(samples.Length * 4);
        foreach (float sample in samples) w.Write(sample);
    }
    private static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    private static void Prepare(string root, string output)
    {
        Directory.CreateDirectory(output);
        // Seed 1729; digest order is stable across runtime/library random-number changes.
        var files = Directory.EnumerateFiles(root, "*.flac", SearchOption.AllDirectories)
            .OrderBy(p => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("1729:" + Path.GetFileName(p)))), StringComparer.Ordinal);
        var selected = new List<(string Path, string Speaker, float[] Pcm)>(); var seen = new HashSet<string>();
        foreach (string file in files)
        {
            string speaker = Path.GetFileName(file).Split('-')[0]; if (seen.Contains(speaker)) continue;
            var pcm = Load(file); if (pcm.Length < 5 * 16000 || pcm.Length > 15 * 16000) continue;
            selected.Add((file, speaker, pcm)); seen.Add(speaker); if (selected.Count == 24) break;
        }
        if (selected.Count != 24) throw new InvalidDataException("Need 24 distinct speakers with 5–15 second utterances.");
        using var manifest = new StreamWriter(Path.Combine(output, "manifest.tsv")); using var sources = new StreamWriter(Path.Combine(output, "sources.tsv"));
        sources.WriteLine("LibriSpeech test-clean; https://www.openslr.org/12/; CC-BY-4.0; Panayotov et al.; seed 1729; unmodified source audio, concatenated without added gaps.");
        for (int i = 0; i < selected.Count; i++) sources.WriteLine(selected[i].Path + "\t" + Hash(selected[i].Path));
        for (int i = 0; i < 12; i++)
        {
            Add("single-" + i.ToString("00", CultureInfo.InvariantCulture), [selected[i]]);
            Add("pair-" + i.ToString("00", CultureInfo.InvariantCulture), i % 2 == 0
                ? [selected[i], selected[i + 12]] : [selected[i], selected[i + 12], selected[i], selected[i + 12]]);
        }
        void Add(string id, (string Path, string Speaker, float[] Pcm)[] clips)
        {
            string wave = Path.Combine(output, id + ".wav"), truth = Path.Combine(output, id + ".rttm");
            Wave(wave, clips.SelectMany(c => c.Pcm).ToArray());
            using var rttm = new StreamWriter(truth); double offset = 0;
            foreach (var clip in clips) { double seconds = clip.Pcm.Length / 16000.0; rttm.WriteLine(FormattableString.Invariant($"SPEAKER {id} 1 {offset:F6} {seconds:F6} <NA> <NA> {clip.Speaker} <NA> <NA>")); offset += seconds; }
            manifest.WriteLine(id + "\t" + wave + "\t" + truth); sources.WriteLine(id + "\t" + Hash(wave) + "\t" + Hash(truth));
        }
    }
    private static void PrepareDev(string root, string output)
    {
        Directory.CreateDirectory(output); using var manifest = new StreamWriter(Path.Combine(output, "manifest.tsv"));
        foreach (string meeting in new[] { "ES2011a", "ES2011b" })
        {
            var entry = DiarizationRun.ReadEntry(root, meeting); DiarizationRun.VerifyEntry(root, entry);
            var pcm = Load(Path.Combine(root, entry.Audio)); using var r = File.OpenRead(Path.Combine(root, entry.Reference)); var reference = Rttm.Read(r);
            for (int i = 0; i < 12; i++)
            {
                // Fixed 10/20/30/60-second excerpts, spread over development only, not selected by outcomes.
                int seconds = i % 4 == 3 ? 60 : 10 * (i % 4 + 1), start = 60 + i * 60; string id = meeting + "-" + i.ToString("00", CultureInfo.InvariantCulture);
                string wave = Path.Combine(output, id + ".wav"), truth = Path.Combine(output, id + ".rttm");
                Wave(wave, pcm.AsSpan(start * 16000, seconds * 16000).ToArray());
                using (var w = new StreamWriter(truth)) foreach (var s in reference.Where(s => s.End > start && s.Start < start + seconds))
                { double a = Math.Max(s.Start, start) - start, b = Math.Min(s.End, start + seconds) - start; w.WriteLine(FormattableString.Invariant($"SPEAKER {id} 1 {a:F6} {b-a:F6} <NA> <NA> {s.Speaker} <NA> <NA>")); }
                manifest.WriteLine(id + "\t" + wave + "\t" + truth);
            }
        }
    }
}
