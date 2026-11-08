using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using System.Xml;
using System.Xml.Linq;

namespace OneSpeech.Evaluation;

internal static class SpeechExcerpt
{
    internal static void Run(string root, string id, string output, string model, bool scoreOnly = false)
    {
        if (id is not ("ES2004a" or "ES2004b")) throw new ArgumentException("Expected a fixed AMI test meeting.");
        var entry = DiarizationRun.ReadEntry(root, id); DiarizationRun.VerifyEntry(root, entry);
        Directory.CreateDirectory(output);
        string crop = Path.Combine(output, id + "-first300s.wav");
        DiarizationExcerpt.Crop(Path.Combine(root, entry.Audio), crop);
        var transcript = scoreOnly ? Transcript.FromJson(File.ReadAllText(Path.Combine(output, "transcript.json")))
            : Speech.Transcribe(crop, CancellationToken.None, language: "en", model: model);
        File.WriteAllText(Path.Combine(output, "transcript.json"), transcript.ToJson());
        File.WriteAllText(Path.Combine(output, "hypothesis.txt"), transcript.Text);
        var words = new List<(double Start, string Speaker, string Text)>();
        using var archive = ZipFile.OpenRead(Path.Combine(root, "downloads", "ami_public_manual_1.6.2.zip"));
        foreach (var file in archive.Entries.Where(e => e.FullName.StartsWith("words/" + id + ".", StringComparison.Ordinal) && e.FullName.EndsWith(".words.xml", StringComparison.Ordinal)))
        {
            using var input = file.Open();
            using var reader = XmlReader.Create(input, new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, MaxCharactersInDocument = 16 * 1024 * 1024 });
            var xml = XDocument.Load(reader);
            foreach (var word in xml.Descendants("w"))
            {
                if (!double.TryParse((string?)word.Attribute("starttime"), NumberStyles.Float, CultureInfo.InvariantCulture, out double start) || start < 0 || start >= 300) continue;
                if (word.Attribute("endtime") is null) continue;
                words.Add((start, Path.GetFileName(file.FullName).Split('.')[1], word.Value));
            }
        }
        string truth = string.Join(' ', words.OrderBy(w => w.Start).ThenBy(w => w.Speaker, StringComparer.Ordinal).Select(w => w.Text));
        File.WriteAllText(Path.Combine(output, "truth.txt"), truth);
        var wer = Metrics.Score(Normalizer.Normalize(truth, "en"), Normalizer.Normalize(transcript.Text, "en"));
        using var referenceInput = File.OpenRead(Path.Combine(root, entry.Reference)); var reference = Rttm.Read(referenceInput);
        string[] speakers = reference.Select(t => t.Speaker).Distinct().Order(StringComparer.Ordinal).ToArray();
        string[] hypotheses = transcript.Words.Where(w => w.Speaker is not null && w.Speaker.Id != SpeakerLabel.Unassigned.Id).Select(w => w.Speaker!.Id).Distinct().Order(StringComparer.Ordinal).ToArray();
        int size = Math.Max(hypotheses.Length, speakers.Length);
        var weights = new double[size, size];
        int eligible = 0, unassigned = 0, overlap = 0, untimed = 0;
        foreach (var word in transcript.Words)
        {
            if (word.Overlap) overlap++;
            bool assigned = word.Speaker is not null && word.Speaker.Id != SpeakerLabel.Unassigned.Id;
            if (!assigned) unassigned++;
            if (word.Start is null || word.End is null) { untimed++; continue; }
            double a = word.Start.Value.TotalSeconds, b = word.End.Value.TotalSeconds;
            var coverage = speakers.Select(s => reference.Where(t => t.Speaker == s).Sum(t => Math.Max(0, Math.Min(t.End, b) - Math.Max(t.Start, a)))).ToArray();
            if (coverage.Length == 0 || coverage.Max() <= 0) continue;
            eligible++;
            if (assigned) weights[Array.IndexOf(hypotheses, word.Speaker!.Id), Array.IndexOf(coverage, coverage.Max())]++;
        }
        double correct = DiarizationMetrics.MaximumAssignment(weights, CancellationToken.None);
        using var outputStream = File.Create(Path.Combine(output, "summary.json")); using var json = new Utf8JsonWriter(outputStream, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("meeting", id); json.WriteString("sourceAudioSha256", entry.AudioHash); json.WriteString("excerptSha256", DiarizationRun.HashFile(crop)); json.WriteString("model", model); json.WriteNumber("durationSeconds", transcript.Duration.TotalSeconds);
        json.WriteString("normalizer", "en/1"); json.WriteNumber("wer", wer.Rate ?? 0); json.WriteNumber("referenceWords", wer.Reference); json.WriteNumber("words", transcript.Words.Length); json.WriteNumber("unassigned", unassigned); json.WriteNumber("overlap", overlap); json.WriteNumber("untimed", untimed);
        json.WriteNumber("eligibleWords", eligible); json.WriteNumber("correctSpeakerWords", correct); json.WriteNumber("correctSpeakerFractionAllWords", transcript.Words.Length == 0 ? 0 : correct / transcript.Words.Length); json.WriteNumber("correctSpeakerFractionEligibleWords", eligible == 0 ? 0 : correct / eligible);
        json.WriteString("speakerMetric", "One-to-one optimal mapping maximizing predicted-word counts; reference label is maximum temporal overlap (ordinal tie); all-word denominator includes unassigned, untimed and no-reference words. Not lexical correctness.");
        json.WriteString("transcript", Path.Combine(output, "transcript.json")); json.WriteEndObject();
        Console.WriteLine("AMI speech excerpt complete: " + output);
    }
}
