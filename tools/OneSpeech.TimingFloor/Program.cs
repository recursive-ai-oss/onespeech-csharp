using System.Globalization;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Evaluation;
using OneSpeech.Whisper;

if (args is ["native", var method, var split, var models, var beginText, var countText])
{
    Environment.SetEnvironmentVariable("ONESPEECH_MODEL_CACHE", models);
    int begin = int.Parse(beginText, CultureInfo.InvariantCulture), count = int.Parse(countText, CultureInfo.InvariantCulture);
    using var whisper = method == "dtw" ? WhisperRecognizer.Load(Path.Combine(models, "whisper-small"), new WhisperOptions { Threads = 4, Temperatures = [0], WordTimestamps = true }) : null;
    if (method is not ("ctc" or "dtw")) throw new ArgumentException("Expected ctc or dtw.");
    string output = Path.Combine(split, method); Directory.CreateDirectory(output);
    foreach (string line in File.ReadLines(Path.Combine(split, "manifest.jsonl")).Skip(begin).Take(count))
    {
        using var doc = JsonDocument.Parse(line); var root = doc.RootElement;
        string id = root.GetProperty("id").GetString()!, target = Path.Combine(output, id + ".json");
        if (File.Exists(target)) continue;
        string path = root.GetProperty("audio").GetString()!;
        Transcript transcript;
        if (whisper is null) transcript = Speech.Align(path, root.GetProperty("text").GetString()!, language: "en", model: "facebook/omniASR-CTC-300M");
        else
        {
            using var input = AudioFileSource.Open(path, CancellationToken.None);
            var samples = new List<float>(); var block = new float[4096]; int n;
            while ((n = input.Read(block, CancellationToken.None)) != 0) samples.AddRange(block.AsSpan(0, n).ToArray());
            transcript = whisper.Transcribe(new AudioBuffer(samples.ToArray(), 16000), "en", CancellationToken.None);
        }
        File.WriteAllText(target + ".part", transcript.ToJson()); File.Move(target + ".part", target);
        Console.WriteLine(id);
    }
    return 0;
}
if (args is ["export", var splitPath])
{
    var truth = new List<Utterance>(); var ctc = new List<Utterance>(); var dtw = new List<Utterance>(); var oracle = new List<Utterance>();
    foreach (string line in File.ReadLines(Path.Combine(splitPath, "manifest.jsonl")))
    {
        using var manifest = JsonDocument.Parse(line); var row = manifest.RootElement;
        string id = row.GetProperty("id").GetString()!, text = row.GetProperty("text").GetString()!;
        string truthPath = Directory.GetFiles(Path.Combine(splitPath, "truth"), id + ".json", SearchOption.AllDirectories).Single();
        using var mfa = JsonDocument.Parse(File.ReadAllText(truthPath));
        var words = mfa.RootElement.GetProperty("tiers").GetProperty("words").GetProperty("entries").EnumerateArray()
            .Where(e => !string.IsNullOrWhiteSpace(e[2].GetString()) && e[2].GetString() != "sil")
            .Select(e => new EvalWord(e[2].GetString()!, e[0].GetDouble(), e[1].GetDouble())).ToArray();
        truth.Add(new(id, text, words));
        foreach (var (name, list) in new[] { ("ctc", ctc), ("dtw", dtw) })
        {
            var t = Transcript.FromJson(File.ReadAllText(Path.Combine(splitPath, name, id + ".json")));
            list.Add(new(id, t.Text, t.Words.Select(w => new EvalWord(w.Text, w.Start?.TotalSeconds, w.End?.TotalSeconds)).ToArray()));
        }
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(splitPath, "oracle", id + ".json")));
        var r = reference.RootElement;
        oracle.Add(new(id, r.GetProperty("text").GetString()!, r.GetProperty("segments").EnumerateArray().SelectMany(s => s.GetProperty("words").EnumerateArray()).Select(w => new EvalWord(w.GetProperty("word").GetString()!, w.GetProperty("start").GetDouble(), w.GetProperty("end").GetDouble())).ToArray()));
    }
    foreach (var (name, list) in new[] { ("truth", truth), ("ctc", ctc), ("dtw", dtw), ("oracle", oracle) }) Hypotheses.Write(Path.Combine(splitPath, name + ".jsonl"), list);
    Summary.Write(splitPath);
    return 0;
}
throw new ArgumentException("Expected native <ctc|dtw> <split> <models> <begin> <count>, or export <split>.");
