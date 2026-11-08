using System.Globalization;
using System.Text.Json;
using OneSpeech.Speakers;

internal static class SpeakerVerification
{
    internal readonly record struct Trial(string First, string Second, bool Same);
    internal readonly record struct Score(bool Same, double Ours, double Reference);
    internal readonly record struct Metrics(double Eer, double MinDcf);
    internal static void Generate(string clean, string other, string output, int seed)
    {
        var ids = Directory.EnumerateFiles(clean, "*.flac", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(other, "*.flac", SearchOption.AllDirectories)).Select(Path.GetFileNameWithoutExtension).Select(x => x!)
            .Order(StringComparer.Ordinal).ToArray();
        var speakers = ids.GroupBy(x => x.Split('-')[0], StringComparer.Ordinal).OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(x => x.Take(20).ToArray()).Where(x => x.Length >= 2).ToArray();
        var trials = MakeTrials(speakers, 10000, seed);
        using var writer = new StreamWriter(output);
        writer.WriteLine($"# seed={seed} pairsPerClass=10000 maxUtterancesPerSpeaker=20; LibriSpeech test-clean+test-other IDs only");
        foreach (var trial in trials) writer.WriteLine($"{(trial.Same ? 1 : 0)}\t{trial.First}\t{trial.Second}");
        Console.WriteLine($"speakers={speakers.Length} utterances={speakers.Sum(x => x.Length)} trials={trials.Length}");
    }
    internal static Trial[] MakeTrials(string[][] speakers, int perClass, int seed)
    {
        if (speakers.Length < 2) throw new ArgumentException("At least two speakers are required.", nameof(speakers));
        var random = new Random(seed);
        var result = new List<Trial>();
        var used = new HashSet<(string, string)>();
        for (int label = 1; label >= 0; label--)
        {
            int count = 0, attempts = 0;
            while (count < perClass)
            {
                if (++attempts > perClass * 1000) throw new InvalidDataException("Insufficient distinct trial pairs.");
                int firstSpeaker = random.Next(speakers.Length), secondSpeaker = firstSpeaker;
                if (label == 0) { secondSpeaker = random.Next(speakers.Length - 1); if (secondSpeaker >= firstSpeaker) secondSpeaker++; }
                var a = speakers[firstSpeaker]; var b = speakers[secondSpeaker];
                string first = a[random.Next(a.Length)], second = b[random.Next(b.Length)];
                int order = StringComparer.Ordinal.Compare(first, second);
                if (order == 0) continue;
                if (order > 0) (first, second) = (second, first);
                if (used.Add((first, second))) { result.Add(new(first, second, label == 1)); count++; }
            }
        }
        return result.OrderBy(x => x.First, StringComparer.Ordinal).ThenBy(x => x.Second, StringComparer.Ordinal).ToArray();
    }
    internal static Trial[] ReadTrials(string path) => File.ReadLines(path).Where(x => !x.StartsWith('#') && !string.IsNullOrWhiteSpace(x)).Select(line =>
    {
        var fields = line.Split('\t');
        if (fields.Length != 3 || fields[0] is not ("0" or "1")) throw new InvalidDataException("Malformed trial.");
        foreach (string id in fields.Skip(1)) if (id.Any(c => !char.IsAsciiDigit(c) && c != '-') || id.Split('-').Length != 3) throw new InvalidDataException("Malformed utterance ID.");
        bool same = fields[1].Split('-')[0] == fields[2].Split('-')[0];
        if (same != (fields[0] == "1") || fields[1] == fields[2]) throw new InvalidDataException("Trial identity/label mismatch.");
        return new Trial(fields[1], fields[2], same);
    }).ToArray();

    internal static Metrics Measure((bool Same, double Value)[] scores)
    {
        int positives = scores.Count(x => x.Same), negatives = scores.Length - positives;
        if (positives == 0 || negatives == 0 || scores.Any(x => !double.IsFinite(x.Value))) throw new ArgumentException("Finite scores and both classes are required.", nameof(scores));
        var ordered = scores.OrderByDescending(x => x.Value).ToArray();
        int acceptedSame = 0, acceptedOther = 0;
        double previousMiss = 1, previousFa = 0, eer = 1, minimum = 1;
        for (int i = 0; i < ordered.Length;)
        {
            double threshold = ordered[i].Value;
            do { if (ordered[i].Same) acceptedSame++; else acceptedOther++; i++; } while (i < ordered.Length && ordered[i].Value == threshold);
            double miss = 1 - (double)acceptedSame / positives, fa = (double)acceptedOther / negatives;
            minimum = Math.Min(minimum, miss + 99 * fa); // normalized DCF: p=.01, Cmiss=Cfa=1, divide by p.
            if (previousFa - previousMiss <= 0 && fa - miss >= 0)
            {
                double weight = (previousMiss - previousFa) / ((fa - miss) - (previousFa - previousMiss));
                eer = previousFa + weight * (fa - previousFa);
            }
            previousFa = fa; previousMiss = miss;
        }
        return new(eer, minimum);
    }
    internal static void ScoreFiles(string trialsPath, string ours, string reference, string output)
    {
        var trials = ReadTrials(trialsPath);
        var ownCache = new Dictionary<string, float[]>(StringComparer.Ordinal); var refCache = new Dictionary<string, float[]>(StringComparer.Ordinal);
        float[] Load(Dictionary<string, float[]> cache, string directory, string id)
        {
            if (cache.TryGetValue(id, out var data)) return data;
            byte[] bytes = File.ReadAllBytes(Path.Combine(directory, id + ".f32"));
            if (bytes.Length is < 4 or > 8192 || bytes.Length % 4 != 0) throw new InvalidDataException("Embedding size bound.");
            data = new float[bytes.Length / 4];
            for (int i = 0; i < data.Length; i++) data[i] = System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
            cache.Add(id, data); return data;
        }
        var scores = trials.Select(t => new Score(t.Same,
            EmbeddingMath.Cosine(Load(ownCache, ours, t.First), Load(ownCache, ours, t.Second)),
            EmbeddingMath.Cosine(Load(refCache, reference, t.First), Load(refCache, reference, t.Second)))).ToArray();
        var ownMetric = Measure(scores.Select(s => (s.Same, s.Ours)).ToArray());
        var refMetric = Measure(scores.Select(s => (s.Same, s.Reference)).ToArray());
        var positive = scores.Where(x => x.Same).ToArray(); var negative = scores.Where(x => !x.Same).ToArray();
        var random = new Random(1729); double[] eerDeltas = new double[1000], dcfDeltas = new double[1000];
        var ownSample = new (bool, double)[scores.Length]; var refSample = new (bool, double)[scores.Length];
        for (int r = 0; r < 1000; r++)
        {
            for (int i = 0; i < scores.Length; i++)
            {
                var source = i < positive.Length ? positive : negative;
                var score = source[random.Next(source.Length)];
                ownSample[i] = (score.Same, score.Ours); refSample[i] = (score.Same, score.Reference);
            }
            var a = Measure(ownSample); var b = Measure(refSample);
            eerDeltas[r] = (a.Eer - b.Eer) * 100; dcfDeltas[r] = a.MinDcf - b.MinDcf;
        }
        Array.Sort(eerDeltas); Array.Sort(dcfDeltas);
        using var file = File.Create(output); using var writer = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject(); writer.WriteNumber("trials", scores.Length); writer.WriteNumber("positiveTrials", positive.Length);
        writer.WriteNumber("pTarget", .01); writer.WriteNumber("cMiss", 1); writer.WriteNumber("cFa", 1); writer.WriteBoolean("dcfNormalized", true);
        writer.WriteNumber("ourEerPercent", ownMetric.Eer * 100); writer.WriteNumber("referenceEerPercent", refMetric.Eer * 100);
        writer.WriteNumber("ourMinDcf", ownMetric.MinDcf); writer.WriteNumber("referenceMinDcf", refMetric.MinDcf);
        writer.WriteNumber("eerDeltaPoints", (ownMetric.Eer - refMetric.Eer) * 100); writer.WriteNumber("minDcfDelta", ownMetric.MinDcf - refMetric.MinDcf);
        writer.WriteNumber("bootstrapSeed", 1729); writer.WriteNumber("bootstrapReplicates", 1000);
        writer.WriteString("resamplingUnit", "paired trial, stratified by same/different; conditional on the fixed utterance/speaker set");
        writer.WriteStartArray("eerDelta95CiPoints"); writer.WriteNumberValue(Quantile(eerDeltas, .025)); writer.WriteNumberValue(Quantile(eerDeltas, .975)); writer.WriteEndArray();
        writer.WriteStartArray("minDcfDelta95Ci"); writer.WriteNumberValue(Quantile(dcfDeltas, .025)); writer.WriteNumberValue(Quantile(dcfDeltas, .975)); writer.WriteEndArray();
        writer.WriteEndObject();
    }
    private static double Quantile(double[] data, double p) { double position = (data.Length - 1) * p; int lo = (int)position; return data[lo] + (data[(int)Math.Ceiling(position)] - data[lo]) * (position - lo); }
}
