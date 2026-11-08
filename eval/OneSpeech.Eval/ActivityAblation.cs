using System.Collections.Immutable;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneSpeech.Audio;
using OneSpeech.Diarization;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

// Development-only activity ablations. No speaker labels from truth reach the embedder or clusterer.
internal static partial class ActivityAblation
{
    private sealed record Setting(float Threshold, int Speech, int Silence, int Pad, int Cutoff = 100)
    {
        internal string Id => FormattableString.Invariant($"T{Threshold * 100:00}-M{Speech}-S{Silence}-P{Pad}-C{Cutoff}");
        internal SileroVadOptions Options => new() { Threshold = Threshold, NegativeThreshold = Math.Max(0.01f, Threshold - 0.15f), MinimumSpeech = TimeSpan.FromMilliseconds(Speech), MinimumSilence = TimeSpan.FromMilliseconds(Silence), SpeechPadding = TimeSpan.FromMilliseconds(Pad) };
    }
    private static readonly float[] Thresholds = [0.2f, 0.35f, 0.5f];
    private static readonly int[] SpeechDurations = [0, 250], SilenceDurations = [100, 250], Pads = [30, 150];
    private static readonly Setting[] Settings = [.. from t in Thresholds from m in SpeechDurations from s in SilenceDurations from p in Pads select new Setting(t, m, s, p)];
    private static readonly Setting Baseline = new(0.5f, 250, 100, 30);
    private static Setting Find(string id) => Settings.Concat(Settings.Select(s => s with { Cutoff = 45 })).Single(s => s.Id == id);

    internal static int Run(string[] args)
    {
        if (args.Length > 0 && args[0].StartsWith("refine-", StringComparison.Ordinal)) return Refinement(args);
        if (args is ["shortlist", var results]) { Select(results, "sad.jsonl", "sadErrorSeconds", "shortlist.txt", 3); return 0; }
        if (args is ["freeze", var resultsRoot]) { Select(resultsRoot, "cluster.jsonl", "errorSeconds", "frozen.txt", 1); return 0; }
        if (args is ["grid", var root, var id, var output]) { Dev(root, id); Grid(root, id, output); return 0; }
        if (args is ["cluster", var graph, var data, var meeting, var destination]) { Dev(data, meeting); Cluster(graph, data, meeting, destination, null); return 0; }
        if (args is ["holdout", var weights, var dataRoot, var testId, var target, var frozen])
        {
            var entry = DiarizationRun.ReadEntry(dataRoot, testId);
            if (entry.Split == "dev" || testId is not ("ES2004a" or "ES2004b" or "R8001_M8004_MS801" or "R8003_M8001_MS801")) throw new ArgumentException("Expected a fixed holdout.");
            Cluster(weights, dataRoot, testId, target, frozen); return 0;
        }
        throw new ArgumentException("activity grid ROOT DEV_ID RESULTS | shortlist RESULTS | cluster GRAPH ROOT DEV_ID RESULTS | freeze RESULTS | holdout GRAPH ROOT ID RESULTS FROZEN_FILE");
    }
    private static void Dev(string root, string id)
    {
        if (id is not ("ES2011a" or "ES2011b") || DiarizationRun.ReadEntry(root, id).Split != "dev") throw new ArgumentException("Activity fitting is restricted to AMI development ES2011a/b.");
    }
    private static List<AudioBuffer> Load(string root, string id)
    {
        var entry = DiarizationRun.ReadEntry(root, id); DiarizationRun.VerifyEntry(root, entry);
        using var source = AudioFileSource.Open(Path.Combine(root, entry.Audio), CancellationToken.None);
        return DiarizationBlocks.Fixed(DiarizationRun.Blocks(source), CancellationToken.None).Select(b => new AudioBuffer(b.Samples.ToArray(), 16000, b.Channels)).ToList();
    }
    private static IReadOnlyList<DiarizationSegment> Reference(string root, string id)
    { using var input = File.OpenRead(Path.Combine(root, DiarizationRun.ReadEntry(root, id).Reference)); return Rttm.Read(input); }
    private static void Grid(string root, string id, string output)
    {
        string directory = Path.Combine(output, id); Directory.CreateDirectory(directory);
        var audio = Load(root, id); var reference = Reference(root, id);
        using var silero = new SileroVad(); var probabilities = new ProbabilityVad(silero, audio);
        using var writer = new StreamWriter(Path.Combine(directory, "sad.jsonl"), false, new UTF8Encoding(false));
        foreach (var setting in Settings)
        {
            probabilities.Options = setting.Options; probabilities.Reset();
            var sad = Intervals(audio, probabilities, id);
            var union = Union(reference, id);
            var score = DiarizationMetrics.Score(union, sad);
            Write(writer, root, id, setting.Id, "sad", score, probabilities.DroppedSeconds, probabilities.DroppedRegions, null, null, null);
        }
        Console.WriteLine(id + " activity grid complete");
    }
    private static void Cluster(string graph, string root, string id, string output, string? frozen)
    {
        string directory = Path.Combine(output, id); Directory.CreateDirectory(directory);
        string scores = Path.Combine(directory, frozen is null ? "cluster.jsonl" : "holdout.jsonl");
        if (File.Exists(scores)) throw new IOException("Scores already exist; never overwrite a held-out measurement.");
        var settings = frozen is null ? File.ReadAllLines(Path.Combine(output, "shortlist.txt")).Append(Baseline.Id).Distinct().Select(Find).Append(Baseline with { Cutoff = 45 }).ToList() : [Find(File.ReadAllText(frozen).Trim())];
        var audio = Load(root, id); var reference = Reference(root, id);
        using var silero = new SileroVad();
        var probabilities = new ProbabilityVad(silero, audio);
        using var actual = DiarizationRun.LoadEmbedder(graph); using var embedder = new CachedEmbedder(actual);
        using var writer = new StreamWriter(scores, false, new UTF8Encoding(false));
        foreach (var setting in settings.DistinctBy(s => s.Id)) Run(setting.Id, setting, probabilities);
        if (frozen is null)
        {
            Run("oracle-C100", Baseline, new OracleVad(reference, silero.Model));
            Run("oracle-C45", Baseline with { Cutoff = 45 }, new OracleVad(reference, silero.Model));
        }
        void Run(string name, Setting setting, ActivityVad vad)
        {
            vad.Options = setting.Options; vad.MinimumSeconds = setting.Cutoff / 1000.0; vad.Reset();
            var options = new ClusteringOptions { MinimumSegmentSeconds = setting.Cutoff / 1000.0 };
            var diarizer = new ClusteringDiarizer(embedder, vad, options);
            var turns = diarizer.DiarizeBlocks(audio, CancellationToken.None);
            string hyp = Path.Combine(directory, name + ".hyp.rttm"); DiarizationRun.WriteTurns(hyp, id, turns);
            using var input = File.OpenRead(hyp);
            var score = DiarizationMetrics.Score(reference, Rttm.Read(input));
            Write(writer, root, id, name, "cluster", score, vad.DroppedSeconds, vad.DroppedRegions, graph, hyp, frozen);
            Console.WriteLine(FormattableString.Invariant($"{id} {name} DER={score.Der * 100:F4} JER={score.Jer * 100:F4} M/F/C={score.MissSeconds:F2}/{score.FalseAlarmSeconds:F2}/{score.ConfusionSeconds:F2} cache={embedder.Count}"));
        }
    }
    private static void Select(string root, string filename, string error, string target, int take)
    {
        var sums = new Dictionary<string, (double Error, double Reference, double Jer)>();
        var seen = new Dictionary<string, HashSet<string>>();
        foreach (string id in new[] { "ES2011a", "ES2011b" })
            foreach (string line in File.ReadLines(Path.Combine(root, id, filename)))
            {
                using var doc = JsonDocument.Parse(line); var e = doc.RootElement;
                string name = e.GetProperty("setting").GetString()!;
                if (name.StartsWith("oracle", StringComparison.Ordinal)) continue;
                if (e.GetProperty("id").GetString() != id) throw new InvalidDataException("Development ID mismatch.");
                sums.TryGetValue(name, out var old);
                sums[name] = (old.Error + e.GetProperty(error).GetDouble(), old.Reference + e.GetProperty("referenceSeconds").GetDouble(), old.Jer + e.GetProperty("jaccardErrorSum").GetDouble());
                if (!seen.TryGetValue(name, out var ids)) seen[name] = ids = [];
                if (!ids.Add(id)) throw new InvalidDataException("Duplicate development row.");
            }
        var ranking = sums.Where(p => seen[p.Key].Count == 2).OrderBy(p => p.Value.Error / p.Value.Reference).ThenBy(p => p.Value.Jer).ThenBy(p => p.Key, StringComparer.Ordinal).ToArray();
        if (ranking.Length < take) throw new InvalidDataException("Insufficient paired development profiles.");
        foreach (var row in ranking) Console.WriteLine(FormattableString.Invariant($"{row.Key}: {100 * row.Value.Error / row.Value.Reference:F5}%"));
        string path = Path.Combine(root, target);
        if (File.Exists(path)) throw new IOException("Selection already frozen.");
        File.WriteAllLines(path, ranking.Take(take).Select(p => p.Key));
        Console.WriteLine("selection SHA256 " + DiarizationRun.HashFile(path));
    }
    private static void Write(StreamWriter writer, string root, string id, string setting, string phase, DiarizationScore score, double dropped, int regions, string? graph, string? hypothesis, string? frozen)
    {
        using var buffer = new MemoryStream(); using (var json = new Utf8JsonWriter(buffer))
        {
            var entry = DiarizationRun.ReadEntry(root, id);
            json.WriteStartObject(); json.WriteString("id", id); json.WriteString("phase", phase); json.WriteString("setting", setting);
            json.WriteString("audioSha256", entry.AudioHash); json.WriteString("referenceSha256", entry.ReferenceHash);
            json.WriteNumber("referenceSeconds", score.ReferenceSeconds); json.WriteNumber("missSeconds", score.MissSeconds); json.WriteNumber("falseAlarmSeconds", score.FalseAlarmSeconds); json.WriteNumber("confusionSeconds", score.ConfusionSeconds);
            json.WriteNumber("errorSeconds", score.MissSeconds + score.FalseAlarmSeconds + score.ConfusionSeconds); json.WriteNumber("sadErrorSeconds", score.MissSeconds + score.FalseAlarmSeconds);
            json.WriteNumber("jaccardErrorSum", score.JaccardErrorSum); json.WriteNumber("referenceSpeakers", score.ReferenceSpeakers); json.WriteNumber("der", score.Der ?? 0); json.WriteNumber("jer", score.Jer ?? 0);
            json.WriteNumber("shortDroppedSeconds", dropped); json.WriteNumber("shortDroppedRegions", regions);
            if (graph is not null) json.WriteString("graphSha256", DiarizationRun.HashFile(graph));
            if (hypothesis is not null) json.WriteString("hypothesis", hypothesis);
            if (frozen is not null) json.WriteString("selectionSha256", DiarizationRun.HashFile(frozen));
            json.WriteEndObject();
        }
        writer.WriteLine(Encoding.UTF8.GetString(buffer.ToArray())); writer.Flush();
    }
    private static List<DiarizationSegment> Intervals(List<AudioBuffer> audio, ActivityVad vad, string id)
    {
        var result = new List<DiarizationSegment>(); double offset = 0;
        foreach (var block in audio)
        { foreach (var interval in vad.Detect(block, CancellationToken.None)) result.Add(new(id, "1", "sad", offset + interval.Start.TotalSeconds, offset + interval.End.TotalSeconds)); offset += block.Duration.TotalSeconds; }
        return result;
    }
    private static List<DiarizationSegment> Union(IReadOnlyList<DiarizationSegment> reference, string id)
    {
        var result = new List<DiarizationSegment>();
        foreach (var r in reference.OrderBy(s => s.Start))
        { if (result.Count != 0 && r.Start <= result[^1].End) result[^1] = result[^1] with { End = Math.Max(result[^1].End, r.End) }; else result.Add(new(id, "1", "sad", r.Start, r.End)); }
        return result;
    }
    private abstract class ActivityVad : VoiceActivityDetector
    {
        internal SileroVadOptions Options { get; set; } = new();
        internal double MinimumSeconds { get; set; } = 0.1;
        internal double DroppedSeconds { get; private set; }
        internal int DroppedRegions { get; private set; }
        protected double Offset { get; private set; }
        internal void Reset() { Offset = 0; DroppedSeconds = 0; DroppedRegions = 0; }
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
        {
            var intervals = DetectCore(audio, cancellationToken);
            foreach (var interval in intervals)
                if (interval.End - interval.Start < TimeSpan.FromSeconds(MinimumSeconds)) { DroppedSeconds += (interval.End - interval.Start).TotalSeconds; DroppedRegions++; }
            Offset += audio.Duration.TotalSeconds;
            return intervals;
        }
        protected abstract ImmutableArray<SpeechInterval> DetectCore(AudioBuffer audio, CancellationToken token);
    }
    private sealed class ProbabilityVad : ActivityVad
    {
        private readonly Dictionary<string, float[]> probabilities = new(StringComparer.Ordinal);
        private readonly SileroVad model;
        internal ProbabilityVad(SileroVad model, List<AudioBuffer> audio)
        {
            this.model = model;
            foreach (var block in audio)
            {
                if (block.Channels != 1) throw new InvalidDataException("Prepared fitting audio must be mono.");
                using var state = model.CreateState(16000);
                float[] values = new float[(block.FrameCount + 511) / 512]; var frame = new float[512];
                for (int i = 0; i < values.Length; i++)
                { Array.Clear(frame); block.Samples.Span.Slice(i * 512, Math.Min(512, block.FrameCount - i * 512)).CopyTo(frame); values[i] = state.ProcessFrame(frame, CancellationToken.None); }
                probabilities.Add(DiarizationRun.PcmHash(block), values);
            }
        }
        public override ModelInfo Model => model.Model;
        protected override ImmutableArray<SpeechInterval> DetectCore(AudioBuffer audio, CancellationToken token)
        {
            var segmenter = new SpeechSegmenter(16000, Options); var raw = new List<SpeechInterval>();
            foreach (float p in probabilities[DiarizationRun.PcmHash(audio)]) { token.ThrowIfCancellationRequested(); if (segmenter.Push(p) is { } interval) raw.Add(interval); }
            if (segmenter.Finish(audio.FrameCount) is { } final) raw.Add(final);
            return SpeechSegmenter.Pad(raw, audio.Duration, Options.SpeechPadding);
        }
    }
    private sealed class OracleVad(IReadOnlyList<DiarizationSegment> reference, ModelInfo model) : ActivityVad
    {
        public override ModelInfo Model => model;
        protected override ImmutableArray<SpeechInterval> DetectCore(AudioBuffer audio, CancellationToken token)
        {
            var result = ImmutableArray.CreateBuilder<SpeechInterval>();
            foreach (var interval in Union(reference, "sad"))
            {
                token.ThrowIfCancellationRequested(); double a = Math.Max(interval.Start, Offset), b = Math.Min(interval.End, Offset + audio.Duration.TotalSeconds);
                if (b > a) result.Add(new SpeechInterval { Start = TimeSpan.FromSeconds(a - Offset), End = TimeSpan.FromSeconds(b - Offset) });
            }
            return result.ToImmutable();
        }
    }
    private sealed class CachedEmbedder(SpeakerEmbedder inner) : SpeakerEmbedder
    {
        private readonly Dictionary<string, float[]> cache = new(StringComparer.Ordinal);
        internal int Count => cache.Count;
        public override ModelInfo Model => inner.Model;
        public override int Dimension => inner.Dimension;
        public override int SampleRate => inner.SampleRate;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            string key = DiarizationRun.PcmHash(audio);
            if (!cache.TryGetValue(key, out var vector)) { vector = new float[Dimension]; inner.Embed(audio, vector, cancellationToken); cache.Add(key, vector); }
            vector.CopyTo(destination);
        }
        public override void Dispose() { foreach (var value in cache.Values) Array.Clear(value); cache.Clear(); }
    }
}
