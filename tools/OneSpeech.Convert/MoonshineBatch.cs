using System.Text.Json;
using OneSpeech;
using OneSpeech.Moonshine;

internal static class MoonshineBatch
{
    internal static void ValidateSaved(string model, string ours, string reference)
    {
        var tokenizer = new MoonshineTokenizer(model);
        int count = 0, differentIds = 0, differentText = 0, incompleteFrames = 0;
        foreach (string file in Directory.EnumerateFiles(ours, "*.json").Order(StringComparer.Ordinal))
        {
            using var native = JsonDocument.Parse(File.ReadAllBytes(file));
            using var oracle = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, Path.GetFileName(file))));
            int[] ids = native.RootElement.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray();
            int[] expected = oracle.RootElement.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray();
            if (!ids.SequenceEqual(expected)) differentIds++;
            if ((long)Math.Round(native.RootElement.GetProperty("duration").GetDouble() * 16000) % 80 != 0) incompleteFrames++;
            if (tokenizer.Decode(ids) != native.RootElement.GetProperty("text").GetString())
                throw new InvalidDataException("Saved text disagrees with current tokenizer: " + file);
            if (native.RootElement.GetProperty("text").GetString() != oracle.RootElement.GetProperty("text").GetString()) differentText++;
            count++;
        }
        Console.WriteLine($"{count} saved hypotheses: ID differences={differentIds}, text differences={differentText}, incomplete 80-sample frames={incompleteFrames}; current decoder text validated.");
    }

    internal static void Run(string model, string splitRoot, string output, int begin, int count, bool refreshIncomplete = false)
    {
        Directory.CreateDirectory(output);
        using var recognizer = MoonshineRecognizer.Load(model, new MoonshineOptions { Threads = 2 });
        var files = Directory.EnumerateFiles(splitRoot, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Skip(begin).Take(count);
        foreach (string file in files)
        {
            string id = Path.GetFileNameWithoutExtension(file), path = Path.Combine(output, id + ".json");
            if (File.Exists(path))
            {
                if (!refreshIncomplete) continue;
                using var saved = JsonDocument.Parse(File.ReadAllBytes(path));
                if ((long)Math.Round(saved.RootElement.GetProperty("duration").GetDouble() * 16000) % 80 == 0) continue;
                File.Move(path, path + ".before-padding-fix", overwrite: true);
            }
            var clip = OneAudio.Audio.Load(file, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            int[] ids = [];
            recognizer.TokenTrace = value => ids = value;
            var result = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
            using (var stream = File.Create(path + ".part")) using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartObject(); writer.WriteString("id", id); writer.WriteString("text", result.Text);
                writer.WriteStartArray("ids"); foreach (int token in ids) writer.WriteNumberValue(token); writer.WriteEndArray();
                writer.WriteNumber("duration", clip.Samples.Length / 16000d); writer.WriteString("precision", "fp32"); writer.WriteNumber("tokensPerSecond", 6.5); writer.WriteEndObject();
            }
            File.Move(path + ".part", path);
            Console.WriteLine(id);
        }
    }
    internal static void Export(string splitRoot, string ours, string reference, string prefix, int count = 200)
    {
        var truth = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(splitRoot, "*.trans.txt", SearchOption.AllDirectories))
            foreach (string line in File.ReadLines(file)) { int space = line.IndexOf(' ', StringComparison.Ordinal); if (space > 0) truth[line[..space]] = line[(space + 1)..]; }
        var ids = Directory.EnumerateFiles(splitRoot, "*.flac", SearchOption.AllDirectories).Select(Path.GetFileNameWithoutExtension).Order(StringComparer.Ordinal).Take(count).ToArray();
        foreach (var (suffix, directory) in new[] { ("ours", ours), ("reference", reference), ("truth", "") })
        {
            using var stream = new StreamWriter(prefix + "." + suffix + ".jsonl");
            foreach (string? id in ids)
            {
                string text;
                if (suffix == "truth") text = truth[id!];
                else { using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, id + ".json"))); text = document.RootElement.GetProperty("text").GetString()!; }
                using var memory = new MemoryStream();
                using (var json = new Utf8JsonWriter(memory)) { json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", text); json.WriteEndObject(); }
                stream.WriteLine(System.Text.Encoding.UTF8.GetString(memory.ToArray()));
            }
        }
    }
}
