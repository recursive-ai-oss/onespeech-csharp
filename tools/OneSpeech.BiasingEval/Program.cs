using System.Globalization;
using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.Evaluation;
using OneSpeech.FastConformer;
using OneSpeech.Qwen3;
using OneSpeech.Whisper;

using var stop = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
var ct = stop.Token;
if (args is ["prepare", var corpus, var state])
{
    Directory.CreateDirectory(state);
    var pins = File.ReadAllLines("tools/scripts/pins/biasing-earnings22.sha256").Select(line => line.Split("  ", 2, StringSplitOptions.None)).ToDictionary(fields => fields[1], fields => fields[0], StringComparer.Ordinal);
    Verify(corpus, "metadata.csv", pins);
    var rows = File.ReadLines(Path.Combine(corpus, "metadata.csv")).Skip(1).Select(line => line.Split(',')).ToArray();
    if (rows.Any(row => row.Length != 8)) throw new InvalidDataException("Unexpected pinned metadata schema.");
    var files = rows.Select(row => (Id: row[0], Seconds: double.Parse(row[6], CultureInfo.InvariantCulture))).OrderBy(row => row.Seconds).ThenBy(row => row.Id, StringComparer.Ordinal).Take(2).ToArray();
    if (files.Length != 2 || files[0].Id != "4474955" || files[1].Id != "4483046") throw new InvalidDataException("Fixed shortest-file selection changed.");
    File.WriteAllLines(Path.Combine(state, "ids.txt"), files.Select(row => row.Id));
    var references = new List<string>(); var selected = new List<string[]>();
    for (int i = 0; i < files.Length; i++)
    {
        string id = files[i].Id;
        Verify(corpus, "media/" + id + ".mp3", pins); Verify(corpus, "transcripts/nlp_references/" + id + ".nlp", pins);
        var words = ReadRows(corpus, id);
        string reference = string.Join(' ', words.Select(row => row.Word)); references.Add(reference);
        var counts = words.GroupBy(row => row.Word, StringComparer.OrdinalIgnoreCase).ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);
        string[] phrases = words.Where(row => !row.SentenceStart && row.Word.Length >= 5 && row.Word.All(char.IsAsciiLetter) && char.IsAsciiLetterUpper(row.Word[0]) && counts[row.Word] is >= 1 and <= 3)
            .Select(row => row.Word).Distinct(StringComparer.OrdinalIgnoreCase).Order(StringComparer.OrdinalIgnoreCase).Take(12).ToArray();
        if (phrases.Length == 0) throw new InvalidDataException("No eligible contextual terms.");
        if (phrases.Any(term => Normalizer.Normalize(term, "en").Split(' ', StringSplitOptions.RemoveEmptyEntries).Length != 1))
            throw new InvalidDataException("This single-word recall diagnostic requires one normalized token per selected term.");
        selected.Add(phrases);
        string folder = Path.Combine(state, "cases", i.ToString(CultureInfo.InvariantCulture)); Directory.CreateDirectory(folder);
        string wavPath = Path.Combine(folder, "audio.wav");
        if (!File.Exists(wavPath))
        {
            using (var source = OneSpeech.Audio.AudioFileSource.Open(Path.Combine(corpus, "media", id + ".mp3"), ct))
            using (var file = File.Create(wavPath + ".partial"))
            using (var wav = new OneAudio.WavEncoder(file, new OneAudio.AudioFormat(16000, 1)))
            {
                float[] block = new float[16000 * source.Channels], mono = new float[16000]; int frames;
                while ((frames = source.Read(block, ct)) != 0)
                {
                    for (int frame = 0; frame < frames; frame++)
                    {
                        double sum = 0; for (int channel = 0; channel < source.Channels; channel++) sum += block[frame * source.Channels + channel];
                        mono[frame] = (float)(sum / source.Channels);
                    }
                    wav.Write(mono.AsSpan(0, frames));
                }
                wav.Finish();
            }
            File.Move(wavPath + ".partial", wavPath);
        }
        File.WriteAllText(Path.Combine(folder, "reference.txt"), reference); File.WriteAllLines(Path.Combine(folder, "phrases.txt"), phrases);
    }
    for (int i = 0; i < files.Length; i++)
    {
        string folder = Path.Combine(state, "cases", i.ToString(CultureInfo.InvariantCulture));
        var referenceSet = Normalizer.Normalize(references[i], "en").Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet(StringComparer.Ordinal);
        string[] controls = selected[1 - i].Where(term => !referenceSet.Contains(Normalizer.Normalize(term, "en"))).ToArray();
        if (controls.Length == 0) throw new InvalidDataException("No absent-phrase controls.");
        File.WriteAllLines(Path.Combine(folder, "control.txt"), controls);
        using var file = File.Create(Path.Combine(folder, "manifest.json")); using var json = new Utf8JsonWriter(file, new() { Indented = true });
        json.WriteStartObject(); json.WriteString("preparation", "full-recordings/1"); json.WriteString("id", files[i].Id); json.WriteNumber("metadataSeconds", files[i].Seconds);
        json.WriteString("dataset", "Earnings-22@c05ab6fd8b4b627d123c922a22a39e993dd37635"); json.WriteString("terms", "CC-BY-SA-4.0; evaluation only, not redistributed");
        foreach (string name in new[] { "audio.wav", "reference.txt", "phrases.txt", "control.txt" })
        { using var input = File.OpenRead(Path.Combine(folder, name)); json.WriteString(name + "Sha256", Convert.ToHexStringLower(SHA256.HashData(input))); }
        foreach (string name in new[] { "metadata.csv", "media/" + files[i].Id + ".mp3", "transcripts/nlp_references/" + files[i].Id + ".nlp" }) json.WriteString(name + "Sha256", pins[name]);
        json.WriteEndObject(); Console.WriteLine($"Case {i} {files[i].Id}: phrases={selected[i].Length}, controls={controls.Length}, seconds={files[i].Seconds}");
    }
    return;
}
if (args is ["run", var family, var modelDirectory, var root, var caseId, var mode])
{
    string folder = Path.Combine(root, "cases", caseId), output = Path.Combine(folder, family + "-" + mode + ".json");
    if (File.Exists(output)) { Console.WriteLine("Already complete: " + output); return; }
    var load = new LoadProgress(Path.Combine(folder, family + "-" + mode + ".load.txt")); load.Sample("before-load");
    SpeechRecognizer recognizer = family switch
    {
        "whisper" => WhisperRecognizer.Load(modelDirectory, new WhisperOptions { Threads = 4, Temperatures = [0], WordTimestamps = false }, ct),
        "qwen" => Qwen3Recognizer.Load(modelDirectory, new Qwen3Options { Threads = 4 }, ct),
        "parakeet" => ParakeetRecognizer.Load(new(modelDirectory, ParakeetInfo()), threads: 4, cancellationToken: ct),
        _ => throw new ArgumentException("Expected whisper, qwen or parakeet.")
    };
    using var owner = (IDisposable)recognizer;
    string expectedId = family switch { "whisper" => "openai/whisper-small", "qwen" => "Qwen/Qwen3-ASR-0.6B", _ => "nvidia/parakeet-tdt-0.6b-v3" };
    if (recognizer.Model.Id != expectedId) throw new ArgumentException("This fixed diagnostic requires " + expectedId);
    if (mode != "baseline")
    {
        if (mode is not ("phrases" or "control")) throw new ArgumentException("Invalid mode.");
        recognizer = recognizer.WithPhrases(File.ReadAllLines(Path.Combine(folder, mode + ".txt")).Select(t => new Phrase(t)), ct);
    }
    using var detector = new OneSpeech.Vad.SileroVad(cancellationToken: ct);
    var pipeline = new OneSpeech.Pipeline.VadTranscriptionPipeline(recognizer, detector, progress: load);
    var result = pipeline.Transcribe(Path.Combine(folder, "audio.wav"), "en", ct);
    load.Sample("after");
    File.WriteAllText(output + ".partial", result.ToJson()); File.Move(output + ".partial", output);
    Console.WriteLine("Completed: " + output); return;
}
if (args is ["score", var rootDirectory])
{
    using var file = File.Create(Path.Combine(rootDirectory, "scores.jsonl")); using var writer = new StreamWriter(file);
    var counts = new Dictionary<(string Family, string Mode, string Metric), List<ErrorCounts>>();
    foreach (string folder in Directory.GetDirectories(Path.Combine(rootDirectory, "cases")).Order(StringComparer.Ordinal))
        foreach (string scoreFamily in new[] { "whisper", "qwen", "parakeet" })
            foreach (string scoreMode in new[] { "baseline", "phrases", "control" })
            {
                string path = Path.Combine(folder, scoreFamily + "-" + scoreMode + ".json"); if (!File.Exists(path)) continue;
                string[] reference = Normalizer.Normalize(File.ReadAllText(Path.Combine(folder, "reference.txt")), "en").Split(' ', StringSplitOptions.RemoveEmptyEntries);
                string[] hypothesis = Normalizer.Normalize(Transcript.FromJson(File.ReadAllText(path)).Text, "en").Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var alignment = Metrics.Align(reference, hypothesis);
                var terms = File.ReadAllLines(Path.Combine(folder, "phrases.txt")).Select(t => Normalizer.Normalize(t, "en")).ToHashSet(StringComparer.Ordinal);
                var controls = File.ReadAllLines(Path.Combine(folder, "control.txt")).Select(t => Normalizer.Normalize(t, "en")).ToHashSet(StringComparer.Ordinal);
                int eligible = reference.Count(terms.Contains), recalled = alignment.Matches.Count(m => m.Equal && terms.Contains(reference[m.Reference]));
                int alarms = hypothesis.Count(controls.Contains);
                foreach (var value in new[] { ("wer", alignment.Counts), ("phraseMissRate", new ErrorCounts(eligible, 0, eligible - recalled, 0)) })
                {
                    var key = (scoreFamily, scoreMode, value.Item1);
                    if (!counts.TryGetValue(key, out var list)) { list = []; counts.Add(key, list); }
                    list.Add(value.Item2);
                }
                using var buffer = new MemoryStream(); using (var json = new Utf8JsonWriter(buffer))
                {
                    json.WriteStartObject(); json.WriteString("case", Path.GetFileName(folder)); json.WriteString("family", scoreFamily); json.WriteString("mode", scoreMode); json.WriteString("normalizer", "en/1"); json.WriteNumber("referenceWords", reference.Length); json.WriteNumber("errors", alignment.Counts.Errors); json.WriteNumber("substitutions", alignment.Counts.Substitutions); json.WriteNumber("deletions", alignment.Counts.Deletions); json.WriteNumber("insertions", alignment.Counts.Insertions); json.WriteNumber("wer", alignment.Counts.Rate ?? 0); json.WriteNumber("phraseOccurrences", eligible); json.WriteNumber("recalled", recalled); if (eligible > 0) json.WriteNumber("phraseRecall", (double)recalled / eligible); else json.WriteNull("phraseRecall"); json.WriteNumber("controlFalseAlarms", alarms); json.WriteEndObject();
                }
                writer.WriteLine(System.Text.Encoding.UTF8.GetString(buffer.ToArray()));
            }
    using var bootstrapFile = File.Create(Path.Combine(rootDirectory, "bootstrap.jsonl")); using var bootstrapWriter = new StreamWriter(bootstrapFile);
    foreach (string name in new[] { "whisper", "qwen", "parakeet" })
        foreach (string comparison in new[] { "phrases", "control" })
            foreach (string metric in new[] { "wer", "phraseMissRate" })
            {
                if (!counts.TryGetValue((name, "baseline", metric), out var baseline) || !counts.TryGetValue((name, comparison, metric), out var biased) || baseline.Count != 2 || biased.Count != 2) continue;
                var result = Metrics.Bootstrap(baseline, biased, 1000, 1729);
                using var buffer = new MemoryStream(); using (var json = new Utf8JsonWriter(buffer))
                {
                    json.WriteStartObject(); json.WriteString("family", name); json.WriteString("comparison", comparison); json.WriteString("metric", metric); json.WriteNumber("files", 2); json.WriteNumber("resamples", result.Resamples); json.WriteNumber("seed", result.Seed);
                    WriteInterval(json, "baseline", result.First); WriteInterval(json, "biased", result.Second); WriteInterval(json, "baselineMinusBiased", result.Difference); json.WriteEndObject();
                }
                bootstrapWriter.WriteLine(System.Text.Encoding.UTF8.GetString(buffer.ToArray()));
            }
    return;
}
throw new ArgumentException("Use prepare CORPUS STATE; run FAMILY MODEL STATE CASE baseline|phrases|control; score STATE.");

static List<(string Word, bool SentenceStart)> ReadRows(string corpus, string id)
{
    bool start = true; var rows = new List<(string, bool)>();
    foreach (string line in File.ReadLines(Path.Combine(corpus, "transcripts", "nlp_references", id + ".nlp")).Skip(1))
    {
        string[] columns = line.Split('|'); if (columns.Length < 7 || columns[0].Length == 0) continue;
        rows.Add((columns[0], start)); start = columns[4].IndexOfAny(['.', '!', '?']) >= 0;
    }
    return rows;
}
static ModelInfo ParakeetInfo()
{
    var entry = ModelCatalog.Default.Get("nvidia/parakeet-tdt-0.6b-v3");
    return new() { Id = entry.Id, Revision = entry.Revision, Source = entry.Source, Tier = entry.Tier, License = entry.License, Attribution = entry.Attribution,
        Files = entry.Files.Concat(entry.ConvertedFiles).Select(file => new ModelFileInfo { Name = file.Name, Sha256 = file.Sha256 }).ToImmutableArray() };
}

static void Verify(string root, string name, IReadOnlyDictionary<string, string> pins)
{
    using var input = File.OpenRead(Path.Combine(root, name));
    if (!pins.TryGetValue(name, out string? expected) || Convert.ToHexStringLower(SHA256.HashData(input)) != expected)
        throw new InvalidDataException("Earnings-22 input pin mismatch: " + name);
}

static void WriteInterval(Utf8JsonWriter writer, string name, Interval? interval)
{
    if (interval is null) { writer.WriteNull(name); return; }
    writer.WriteStartObject(name); writer.WriteNumber("lower", interval.Lower); writer.WriteNumber("upper", interval.Upper); writer.WriteEndObject();
}

internal sealed class LoadProgress(string path) : IProgress<TimeSpan>
{
    public void Report(TimeSpan value) => Sample("audio-seconds=" + value.TotalSeconds.ToString(CultureInfo.InvariantCulture));
    internal void Sample(string phase)
    {
        if (File.Exists("/proc/loadavg")) File.AppendAllText(path, phase + " " + File.ReadAllText("/proc/loadavg").Trim() + Environment.NewLine);
    }
}
