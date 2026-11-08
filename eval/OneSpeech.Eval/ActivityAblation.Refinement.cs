using System.Globalization;
using System.Text;
using System.Text.Json;
using OneSpeech.Diarization;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

internal static partial class ActivityAblation
{
    private static readonly Dictionary<string, CosineResegmenter?> Refinements = CreateRefinements();
    private static Dictionary<string, CosineResegmenter?> CreateRefinements()
    {
        var result = new Dictionary<string, CosineResegmenter?>(StringComparer.Ordinal) { ["none"] = null };
        foreach (int radius in new[] { 0, 2, 4 }) foreach (int penalty in new[] { 0, 50, 100, 200, 400 })
            result.Add(FormattableString.Invariant($"R{radius}-P{penalty:000}"), new CosineResegmenter(penalty / 1000.0, radius));
        return result;
    }
    private static int Refinement(string[] args)
    {
        if (args is ["refine-freeze", var results])
        {
            string frozen = Path.Combine(results, "refinement.txt");
            if (File.Exists(frozen)) throw new IOException("Refinement selection already frozen.");
            var errors = new Dictionary<string, (double Error, double Jer, int Count)>();
            foreach (string id in new[] { "ES2011a", "ES2011b" })
            {
                var seen = new HashSet<string>();
                foreach (string line in File.ReadLines(Path.Combine(results, id, "refinement.jsonl")))
                {
                    using var doc = JsonDocument.Parse(line); var e = doc.RootElement;
                    string name = e.GetProperty("setting").GetString()!;
                    if (!Refinements.ContainsKey(name) || !seen.Add(name) || e.GetProperty("id").GetString() != id) throw new InvalidDataException("Invalid development row.");
                    errors.TryGetValue(name, out var old);
                    errors[name] = (old.Error + e.GetProperty("errorSeconds").GetDouble(), old.Jer + e.GetProperty("jaccardErrorSum").GetDouble(), old.Count + 1);
                }
                if (seen.Count != Refinements.Count) throw new InvalidDataException("Incomplete development grid.");
            }
            var best = errors.OrderBy(p => p.Value.Error).ThenBy(p => p.Value.Jer).ThenBy(p => p.Key, StringComparer.Ordinal).First();
            string winner = best.Value.Error < errors["none"].Error - 1e-9 ? best.Key : "none";
            File.WriteAllText(frozen, winner + "\n", new UTF8Encoding(false));
            File.WriteAllText(frozen + ".sha256", DiarizationRun.HashFile(frozen) + "\n", new UTF8Encoding(false));
            foreach (var p in errors.OrderBy(p => p.Value.Error)) Console.WriteLine(FormattableString.Invariant($"{p.Key}: errorSeconds={p.Value.Error:F6} jerSum={p.Value.Jer:F6}"));
            Console.WriteLine("frozen " + winner + " SHA256 " + DiarizationRun.HashFile(frozen)); return 0;
        }
        if (args is ["refine-dev", var graph, var root, var devId, var output])
        { Dev(root, devId); Refine(graph, root, devId, output, null); return 0; }
        if (args is ["refine-holdout", var weights, var data, var meeting, var destination, var frozenFile])
        {
            if (meeting is not ("ES2004a" or "ES2004b" or "R8001_M8004_MS801" or "R8003_M8001_MS801") || DiarizationRun.ReadEntry(data, meeting).Split == "dev") throw new ArgumentException("Expected a fixed holdout.");
            if (DiarizationRun.HashFile(frozenFile) != File.ReadAllText(frozenFile + ".sha256").Trim()) throw new InvalidDataException("Frozen settings digest mismatch.");
            Refine(weights, data, meeting, destination, frozenFile); return 0;
        }
        throw new ArgumentException("refine-dev GRAPH ROOT DEV_ID RESULTS | refine-freeze RESULTS | refine-holdout GRAPH ROOT ID RESULTS FROZEN_FILE");
    }
    private static void Refine(string graph, string root, string id, string output, string? frozen)
    {
        var settings = frozen is null ? Refinements.Keys.ToArray() : [File.ReadAllText(frozen).Trim()];
        string directory = Path.Combine(output, id); Directory.CreateDirectory(directory);
        string scores = Path.Combine(directory, frozen is null ? "refinement.jsonl" : "holdout.jsonl");
        if (File.Exists(scores)) throw new IOException("Scores already exist; refusing overwrite.");
        var audio = Load(root, id); var reference = Reference(root, id);
        using var silero = new SileroVad(); var probabilities = new ProbabilityVad(silero, audio);
        using var actual = DiarizationRun.LoadEmbedder(graph); using var embedder = new CachedEmbedder(actual);
        using var writer = new StreamWriter(scores, false, new UTF8Encoding(false));
        foreach (string name in settings)
        {
            probabilities.Options = new Setting(0.2f, 250, 250, 150).Options; probabilities.Reset();
            var diarizer = new ClusteringDiarizer(embedder, probabilities, resegmenter: Refinements[name]);
            var turns = diarizer.DiarizeBlocks(audio, CancellationToken.None);
            string hyp = Path.Combine(directory, name + ".hyp.rttm"); DiarizationRun.WriteTurns(hyp, id, turns);
            using var input = File.OpenRead(hyp); var score = DiarizationMetrics.Score(reference, Rttm.Read(input));
            Write(writer, root, id, name, "refinement", score, probabilities.DroppedSeconds, probabilities.DroppedRegions, graph, hyp, frozen);
            Console.WriteLine(FormattableString.Invariant($"{id} {name} DER={score.Der * 100:F4} JER={score.Jer * 100:F4} M/F/C={score.MissSeconds:F2}/{score.FalseAlarmSeconds:F2}/{score.ConfusionSeconds:F2}"));
        }
    }
}
