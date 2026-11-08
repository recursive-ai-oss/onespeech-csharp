using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

// Worker-only evaluation. Replay caches live only in RAM and are discarded at meeting boundaries.
internal static class DiarizationRun
{
    internal const string GraphHash = "f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c";
    private static readonly int[] LandmarkGrid = [64, 128, 256];
    private static readonly double[] ThresholdGrid = [-0.2, -0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85];
    internal static readonly Profile[] Profiles = [.. LandmarkGrid.SelectMany(l => new[] { new Profile($"S{l}", ClusteringMethod.Spectral, l, 0.65) }.Concat(ThresholdGrid.Select(t => new Profile($"A{l}-{t.ToString("F2", CultureInfo.InvariantCulture)}", ClusteringMethod.Agglomerative, l, t))))];

    internal static int Run(string[] args)
    {
        if (args is ["speech-excerpt" or "speech-excerpt-score", var rootPath, var meeting, var destination, var model]) { SpeechExcerpt.Run(rootPath, meeting, destination, model, args[0] == "speech-excerpt-score"); return 0; }
        if (args.Length == 2 && args[0] == "select") { Select(args[1]); return 0; }
        if (args.Length == 2 && args[0] == "summary") { DiarizationSummary.Run(args[1]); return 0; }
        if (args.Length == 3 && args[0] == "channels") { ChannelCalibration.Run(args[1], args[2]); return 0; }
        if (args.Length == 6 && args[0] == "excerpt") { DiarizationExcerpt.Run(args[1], args[2], args[3], args[4], Profiles.Single(p => p.Name == args[5])); return 0; }
        if (args.Length < 5 || args[0] is not ("dev" or "holdout")) throw new ArgumentException("diarize dev|holdout GRAPH DATA_ROOT MEETING RESULTS_ROOT [PROFILE ...], or diarize select RESULTS_ROOT");
        string phase = args[0], graph = args[1], root = args[2], id = args[3], output = Path.Combine(args[4], phase, id);
        var entry = ReadEntry(root, id);
        if (phase == "dev" && (entry.Split != "dev" || id is not ("ES2011a" or "ES2011b"))) throw new ArgumentException("Development fitting is restricted to AMI ES2011a/ES2011b.");
        if (phase == "holdout" && (entry.Split == "dev" || id is not ("ES2004a" or "ES2004b" or "R8001_M8004_MS801" or "R8003_M8001_MS801"))) throw new ArgumentException("Holdout command requires one of the four fixed held-out meetings.");
        var profiles = phase == "dev" ? Profiles : args.Skip(5).Select(name => Profiles.Single(p => p.Name == name)).ToArray();
        if (profiles.Length == 0) throw new ArgumentException("Holdout requires explicitly frozen profile names.");
        VerifyGraph(graph); VerifyEntry(root, entry);
        Directory.CreateDirectory(output);
        string scorePath = Path.Combine(output, "scores.jsonl");
        if (File.Exists(scorePath)) throw new IOException("Completed scores already exist; use a new result root rather than overwrite a measurement.");
        using var actual = LoadEmbedder(graph);
        using var actualVad = new SileroVad();
        using var cache = new ReplayEmbedder(actual);
        var vad = new ReplayVad(actualVad);
        using var referenceStream = File.OpenRead(Path.Combine(root, entry.Reference)); var reference = Rttm.Read(referenceStream);
        using (var scores = new StreamWriter(scorePath + ".partial", false, new UTF8Encoding(false)))
        {
            foreach (var profile in profiles)
            {
                cache.Reset(); vad.Reset();
                var diarizer = new ClusteringDiarizer(cache, vad, profile.Options);
                using var source = AudioFileSource.Open(Path.Combine(root, entry.Audio), CancellationToken.None);
                var turns = diarizer.DiarizeBlocks(Blocks(source), CancellationToken.None);
                cache.Complete(); vad.Complete();
                string hypothesis = Path.Combine(output, profile.Name + ".hyp.rttm");
                WriteTurns(hypothesis, id, turns);
                using var hypothesisStream = File.OpenRead(hypothesis);
                var score = DiarizationMetrics.Score(reference, Rttm.Read(hypothesisStream));
                scores.WriteLine(ScoreJson(phase, root, entry, profile, score, hypothesis, graph, turns.Select(t => t.Speaker.Id).Distinct().Count(), cache.Count)); scores.Flush();
                Console.WriteLine(FormattableString.Invariant($"{id} {profile.Name}: DER={100 * score.Der:F4}% JER={100 * score.Jer:F4}% windows={cache.Count}"));
            }
        }
        File.Move(scorePath + ".partial", scorePath);
        return 0;
    }

    internal static CampPlusEmbedder LoadEmbedder(string graph)
    {
        var entry = GraphEntry(graph);
        var model = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = entry.Tier, License = entry.License, Attribution = entry.Attribution, Files = [.. entry.Files.Concat(entry.ConvertedFiles).Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 })] };
        return new CampPlusEmbedder(graph, model, dimension: entry.Id.Contains("zh-cn", StringComparison.Ordinal) ? 192 : 512, threads: 8);
    }
    private static CatalogEntry GraphEntry(string graph)
    {
        string hash = HashFile(graph);
        var entry = ModelCatalog.Default.Get(hash == GraphHash
            ? "iic/speech_campplus_sv_en_voxceleb_16k" : "iic/speech_campplus_sv_zh-cn_16k-common");
        if (!entry.ConvertedFiles.Any(f => f.Sha256 == hash)) throw new InvalidDataException("CAM++ converted graph SHA-256 mismatch.");
        return entry;
    }
    internal static void VerifyGraph(string graph) { _ = GraphEntry(graph); }
    internal static string HashFile(string path) { using var input = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(input)); }
    internal static Entry ReadEntry(string root, string id)
    {
        foreach (string line in File.ReadLines(Path.Combine(root, "manifest.jsonl")))
        {
            using var json = JsonDocument.Parse(line); var e = json.RootElement;
            if (e.GetProperty("id").GetString() != id) continue;
            string S(string key) => e.GetProperty(key).GetString() ?? throw new InvalidDataException("Missing manifest value.");
            return new Entry(id, S("split"), S("license"), S("audio"), S("reference"), S("audioSha256"), S("referenceSha256"));
        }
        throw new InvalidDataException("Meeting absent from prepared manifest.");
    }
    internal static void VerifyEntry(string root, Entry entry)
    { if (HashFile(Path.Combine(root, entry.Audio)) != entry.AudioHash || HashFile(Path.Combine(root, entry.Reference)) != entry.ReferenceHash) throw new InvalidDataException("Prepared audio/reference hash mismatch."); }
    internal static IEnumerable<AudioBuffer> Blocks(AudioFileSource source)
    {
        var buffer = new float[16000 * source.Channels]; int count;
        while ((count = source.Read(buffer, CancellationToken.None)) != 0) yield return new AudioBuffer(buffer.AsMemory(0, count * source.Channels), 16000, source.Channels);
    }
    internal static void WriteTurns(string path, string id, ImmutableArray<Turn> turns)
    {
        using (var writer = new StreamWriter(path + ".partial", false, new UTF8Encoding(false)))
            foreach (var turn in turns) writer.WriteLine(FormattableString.Invariant($"SPEAKER {id} 1 {turn.Start.TotalSeconds:R} {(turn.End - turn.Start).TotalSeconds:R} <NA> <NA> {turn.Speaker.Id} <NA> <NA>"));
        File.Move(path + ".partial", path, overwrite: true);
    }
    private static string ScoreJson(string phase, string root, Entry entry, Profile profile, DiarizationScore score, string hyp, string graph, int speakers, int windows)
    {
        using var buffer = new MemoryStream();
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject(); w.WriteString("phase", phase); w.WriteString("id", entry.Id); w.WriteString("split", entry.Split); w.WriteString("license", entry.License);
            w.WriteString("audioSha256", entry.AudioHash); w.WriteString("referenceSha256", entry.ReferenceHash); w.WriteString("manifestSha256", HashFile(Path.Combine(root, "manifest.jsonl")));
            w.WriteString("profile", profile.Name); w.WriteString("method", profile.Method.ToString()); w.WriteNumber("landmarks", profile.Landmarks); w.WriteNumber("ahcThreshold", profile.Threshold);
            w.WriteNumber("windowSeconds", 1.5); w.WriteNumber("shiftSeconds", 0.25); w.WriteNumber("minSpeakers", 1); w.WriteNumber("maxSpeakers", 8);
            w.WriteString("embedder", GraphEntry(graph).Id); w.WriteString("revision", GraphEntry(graph).Revision); w.WriteString("graphSha256", HashFile(graph)); w.WriteString("precision", "fp32"); w.WriteNumber("threads", 8);
            w.WriteString("vad", "Silero-v6.2.3/threshold0.5/negative0.35/minSpeech250ms/minSilence100ms/pad30ms/independent30s-horizons");
            w.WriteNumber("referenceSeconds", score.ReferenceSeconds); w.WriteNumber("missSeconds", score.MissSeconds); w.WriteNumber("falseAlarmSeconds", score.FalseAlarmSeconds); w.WriteNumber("confusionSeconds", score.ConfusionSeconds);
            w.WriteNumber("jaccardErrorSum", score.JaccardErrorSum); w.WriteNumber("referenceSpeakers", score.ReferenceSpeakers);
            if (score.Der is {} der) w.WriteNumber("der", der); else w.WriteNull("der"); if (score.Jer is {} jer) w.WriteNumber("jer", jer); else w.WriteNull("jer");
            w.WriteNumber("hypothesisSpeakers", speakers); w.WriteNumber("embeddingWindows", windows); w.WriteString("hypothesisRttm", hyp); w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.ToArray());
    }
    private static void Select(string root)
    {
        var rows = new List<DevelopmentRow>();
        foreach (string id in new[] { "ES2011a", "ES2011b" })
            foreach (string line in File.ReadLines(Path.Combine(root, "dev", id, "scores.jsonl")))
            {
                using var json = JsonDocument.Parse(line); var e = json.RootElement;
                if (e.GetProperty("phase").GetString() != "dev" || e.GetProperty("graphSha256").GetString() != GraphHash) throw new InvalidDataException("Unexpected development provenance.");
                double D(string name) => e.GetProperty(name).GetDouble();
                rows.Add(new DevelopmentRow(e.GetProperty("profile").GetString()!, id, D("referenceSeconds"), D("missSeconds") + D("falseAlarmSeconds") + D("confusionSeconds"), D("jaccardErrorSum"), e.GetProperty("referenceSpeakers").GetInt32()));
            }
        var ranking = rows.GroupBy(r => r.Profile).Select(g => new { Name = g.Key, Meetings = g.Select(r => r.Id).Distinct().Count(), Der = g.Sum(r => r.Error) / g.Sum(r => r.Reference), Jer = g.Sum(r => r.JerSum) / g.Sum(r => r.Speakers) }).OrderBy(r => r.Der).ThenBy(r => r.Jer).ThenBy(r => r.Name, StringComparer.Ordinal).ToArray();
        if (ranking.Length != Profiles.Length || ranking.Any(r => r.Meetings != 2)) throw new InvalidDataException("Development grid is incomplete.");
        using var output = File.Create(Path.Combine(root, "selection.json")); using var w = new Utf8JsonWriter(output, new JsonWriterOptions { Indented = true });
        w.WriteStartObject(); w.WriteString("policy", "Minimize pooled AMI-dev DER; tie by speaker-macro JER, then ordinal profile ID. No holdout data used."); w.WriteString("selected", ranking[0].Name);
        foreach (var method in new[] { ClusteringMethod.Spectral, ClusteringMethod.Agglomerative })
        { string name = ranking.First(r => Profiles.Single(p => p.Name == r.Name).Method == method).Name; w.WriteString(method.ToString(), name); }
        w.WriteStartArray("ranking"); foreach (var row in ranking) { w.WriteStartObject(); w.WriteString("profile", row.Name); w.WriteNumber("der", row.Der); w.WriteNumber("jer", row.Jer); w.WriteEndObject(); Console.WriteLine(FormattableString.Invariant($"{row.Name}: dev DER={100 * row.Der:F4}% JER={100 * row.Jer:F4}%")); } w.WriteEndArray(); w.WriteEndObject();
    }
    internal sealed record Entry(string Id, string Split, string License, string Audio, string Reference, string AudioHash, string ReferenceHash);
    internal sealed record Profile(string Name, ClusteringMethod Method, int Landmarks, double Threshold)
    { internal ClusteringOptions Options => new() { Method = Method, MaxLandmarks = Landmarks, AgglomerativeThreshold = Threshold, WindowSeconds = 1.5, ShiftSeconds = 0.25, MinSpeakers = 1, MaxSpeakers = 8, MaxWindows = 100000 }; }
    private sealed record DevelopmentRow(string Profile, string Id, double Reference, double Error, double JerSum, int Speakers);

    private static string PcmHash(AudioBuffer audio)
    {
        byte[] bytes = ArrayPool<byte>.Shared.Rent(audio.Samples.Length * 4);
        try { for (int i = 0; i < audio.Samples.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), audio.Samples.Span[i]); return Convert.ToHexStringLower(SHA256.HashData(bytes.AsSpan(0, audio.Samples.Length * 4))); }
        finally { ArrayPool<byte>.Shared.Return(bytes); }
    }
    private sealed class ReplayEmbedder(SpeakerEmbedder inner) : SpeakerEmbedder
    {
        private readonly List<(string Hash, float[] Values)> cache = []; private int cursor;
        internal int Count => cache.Count;
        internal void Reset() => cursor = 0;
        internal void Complete() { if (cursor != cache.Count) throw new InvalidDataException("Embedding window sequence changed during parameter fitting."); }
        public override ModelInfo Model => inner.Model; public override int Dimension => inner.Dimension; public override int SampleRate => inner.SampleRate;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); string hash = PcmHash(audio);
            if (cursor == cache.Count) { var vector = new float[Dimension]; inner.Embed(audio, vector, cancellationToken); cache.Add((hash, vector)); if (cache.Count % 250 == 0) { Console.WriteLine($"embedding windows {cache.Count}"); if (File.Exists("/proc/loadavg")) Console.WriteLine("load " + File.ReadAllText("/proc/loadavg").Trim()); } }
            if (cache[cursor].Hash != hash) throw new InvalidDataException("Embedding input changed between profiles."); cache[cursor++].Values.CopyTo(destination);
        }
        public override void Dispose() { foreach (var item in cache) Array.Clear(item.Values); cache.Clear(); }
    }
    private sealed class ReplayVad(VoiceActivityDetector inner) : VoiceActivityDetector
    {
        private readonly List<(string Hash, ImmutableArray<SpeechInterval> Intervals)> cache = []; private int cursor;
        internal void Reset() => cursor = 0;
        internal void Complete() { if (cursor != cache.Count) throw new InvalidDataException("VAD block sequence changed during parameter fitting."); }
        public override ModelInfo Model => inner.Model;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); string hash = PcmHash(audio);
            if (cursor == cache.Count) cache.Add((hash, inner.Detect(audio, cancellationToken)));
            if (cache[cursor].Hash != hash) throw new InvalidDataException("VAD input changed between profiles."); return cache[cursor++].Intervals;
        }
    }
}
