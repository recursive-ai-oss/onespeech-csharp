using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Alignment;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Evaluation;
using OneSpeech.Pipeline;
using OneSpeech.Qwen3;

// Retiming fixed hypotheses isolates timing policy from recognition and uses the same written words on both sides.
internal static class RetimingProbe
{
    internal static int Run(string manifest, string input, string output, int begin, int count)
    {
        Directory.CreateDirectory(output);
        var store = ModelStore.Default;
        var q = store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B").AsTask().GetAwaiter().GetResult();
        var c = store.PullAsync("facebook/omniASR-CTC-300M").AsTask().GetAwaiter().GetResult();
        using var qwen = Qwen3ForcedAligner.Load(q.Directory, 8);
        using var ctc = new OmniAsrAligner(Path.Combine(c.Directory, "ctc-fp32.onemodel"), Path.Combine(c.Directory, "omniASR_tokenizer.model"), c.Info, 8);
        using var detector = new OneSpeech.Vad.SileroVad();
        foreach (string line in File.ReadLines(manifest).Skip(begin).Take(count))
        {
            using var row = JsonDocument.Parse(line);
            string id = row.RootElement.GetProperty("id").GetString()!, target = Path.Combine(output, id + ".json");
            if (File.Exists(target)) continue;
            var original = Transcript.FromJson(File.ReadAllText(Path.Combine(input, id + ".json")));
            var clip = OneAudio.Audio.Load(row.RootElement.GetProperty("audio").GetString()!, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var chunks = new VadChunker(detector).Read([new AudioBuffer(clip.Samples, 16000)], null, CancellationToken.None).ToArray();
            Transcript result;
            if (chunks.Length != 1)
            {
                // A cached whole-file hypothesis cannot reconstruct its per-chunk written-word ranges.
                // Re-run only these few files rather than guessing which chunk recognized a boundary word.
                result = Speech.Transcribe(row.RootElement.GetProperty("audio").GetString()!, language: "en", model: "openai/whisper-large-v3-turbo");
            }
            else
            {
                var chunk = chunks[0];
                var local = original with { Words = [.. original.Words.Select(w => w with { Start = w.Start - chunk.Start, End = w.End - chunk.Start })] };
                var pipeline = new VadTranscriptionPipeline(new FixedRecognizer(local), new FullVad(original.Models[0])) { TimingAligners = [qwen, ctc] };
                var timed = pipeline.Transcribe(chunk.Audio, "en", CancellationToken.None);
                var words = timed.Words.Select(w => w with { Start = w.Start + chunk.Start, End = w.End + chunk.Start }).ToImmutableArray();
                int offset = 0;
                var segments = original.Segments.Select(s =>
                {
                    var projected = words.Slice(offset, s.Words.Length); offset += s.Words.Length;
                    return s with { Words = projected };
                }).ToImmutableArray();
                if (offset != words.Length) throw new InvalidDataException("Cached segment word partition differs from the document.");
                result = original with { Words = words, Segments = segments, Models = timed.Models };
            }
            if (original.Text != result.Text || !original.Words.Select(w => w.Text).SequenceEqual(result.Words.Select(w => w.Text)))
                throw new InvalidDataException("Retiming changed recognized words: " + id);
            File.WriteAllText(target + ".part", result.ToJson()); File.Move(target + ".part", target);
            Console.WriteLine(id);
        }
        return 0;
    }

    internal static int Score(string manifest, string truthRoot, string before, string after)
    {
        var scores = new Dictionary<string, (List<double> Start, List<double> End)>(StringComparer.Ordinal)
        { ["before"] = ([], []), ["after"] = ([], []) };
        var provenance = new Dictionary<TimingMethod, int>();
        int common = 0, missingBefore = 0, missingAfter = 0;
        foreach (string line in File.ReadLines(manifest))
        {
            using var row = JsonDocument.Parse(line);
            string id = row.RootElement.GetProperty("id").GetString()!;
            var b = Transcript.FromJson(File.ReadAllText(Path.Combine(before, id + ".json")));
            var a = Transcript.FromJson(File.ReadAllText(Path.Combine(after, id + ".json")));
            if (!b.Words.Select(w => w.Text).SequenceEqual(a.Words.Select(w => w.Text))) throw new InvalidDataException("Retiming changed written words: " + id);
            foreach (var word in a.Words) provenance[word.TimingMethod] = provenance.GetValueOrDefault(word.TimingMethod) + 1;
            missingBefore += b.Words.Count(w => w.Start is null); missingAfter += a.Words.Count(w => w.Start is null);
            using var mfa = JsonDocument.Parse(File.ReadAllText(Directory.GetFiles(truthRoot, id + ".json", SearchOption.AllDirectories).Single()));
            var truth = mfa.RootElement.GetProperty("tiers").GetProperty("words").GetProperty("entries").EnumerateArray()
                .Where(e => !string.IsNullOrWhiteSpace(e[2].GetString()) && e[2].GetString() != "sil")
                .Select(e => (Text: Normalizer.Normalize(e[2].GetString()!, "en"), Start: e[0].GetDouble(), End: e[1].GetDouble())).ToArray();
            var matches = Metrics.Align(truth.Select(w => w.Text).ToArray(), a.Words.Select(w => Normalizer.Normalize(w.Text, "en")).ToArray());
            foreach (var match in matches.Matches.Where(m => m.Equal && truth[m.Reference].Text.Length != 0))
            {
                var bw = b.Words[match.Hypothesis]; var aw = a.Words[match.Hypothesis]; var t = truth[match.Reference];
                if (bw.Start is null || bw.End is null || aw.Start is null || aw.End is null) continue;
                common++;
                scores["before"].Start.Add(Math.Abs(bw.Start.Value.TotalSeconds - t.Start) * 1000);
                scores["before"].End.Add(Math.Abs(bw.End.Value.TotalSeconds - t.End) * 1000);
                scores["after"].Start.Add(Math.Abs(aw.Start.Value.TotalSeconds - t.Start) * 1000);
                scores["after"].End.Add(Math.Abs(aw.End.Value.TotalSeconds - t.End) * 1000);
            }
        }
        Console.WriteLine($"common={common} missingBefore={missingBefore} missingAfter={missingAfter}");
        foreach (var (name, score) in scores)
            Console.WriteLine(FormattableString.Invariant($"{name} startMs={score.Start.Average():F4} endMs={score.End.Average():F4} start50={Within(score.Start, 50):F6} start100={Within(score.Start, 100):F6} end50={Within(score.End, 50):F6} end100={Within(score.End, 100):F6}"));
        foreach (var (method, n) in provenance) Console.WriteLine($"{method}={n}");
        return 0;
    }
    private static double Within(List<double> values, int ms) => (double)values.Count(v => v <= ms + 1e-9) / values.Count;
    private sealed class FixedRecognizer(Transcript transcript) : SpeechRecognizer
    {
        public override ModelInfo Model => transcript.Models.First();
        public override Capabilities Capabilities { get; } = new() { Languages = ["en"] };
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken) => transcript;
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => transcript.Segments;
    }
    private sealed class FullVad(ModelInfo model) : VoiceActivityDetector
    {
        public override ModelInfo Model => model;
        public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken) => [new() { Start = TimeSpan.Zero, End = audio.Duration }];
    }
}
