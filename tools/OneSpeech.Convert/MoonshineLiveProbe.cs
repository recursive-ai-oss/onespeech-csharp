using System.Text.Json;
using OneSpeech;
using OneSpeech.Moonshine;
using OneSpeech.Vad;

internal static class MoonshineLiveProbe
{
    internal static void Check(string directory, string path)
    {
        using var recognizer = MoonshineRecognizer.Load(directory);
        var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        int[] expected = [], actual = [];
        recognizer.TokenTrace = ids => expected = ids;
        var whole = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
        recognizer.TokenTrace = null;
        using var state = new MoonshineStreamingState(recognizer);
        for (int end = 1600; end < clip.Samples.Length; end += 1600) state.Decode(clip.Samples.Slice(0, end), "en", false, TimeSpan.Zero, TimeSpan.FromSeconds(end / 16000d), CancellationToken.None);
        recognizer.TokenTrace = ids => actual = ids;
        var final = state.Decode(clip.Samples, "en", true, TimeSpan.Zero, TimeSpan.FromSeconds(clip.Samples.Length / 16000d), CancellationToken.None);
        if (whole.Text != final.Text || !expected.SequenceEqual(actual)) throw new InvalidOperationException("Cached streaming final differs from whole-utterance decoding.");
        using var vad = new SileroVad();
        using var session = recognizer.CreateLiveSession(vad, "en");
        var events = new List<LiveEvent>();
        for (int start = 0; start < clip.Samples.Length; start += 1600) events.AddRange(session.Push(new AudioBuffer(clip.Samples.Slice(start, Math.Min(1600, clip.Samples.Length - start)), 16000)));
        events.AddRange(session.Complete());
        int partials = events.OfType<PartialEvent>().Count(), finals = events.OfType<FinalEvent>().Count();
        if (partials == 0 || finals == 0) throw new InvalidOperationException("Expected true partials and a final.");
        Console.WriteLine($"Cached final token identity: {expected.Length} tokens; session partials={partials} finals={finals}; {final.Text}");
    }

    internal static void Compare(string before, string after)
    {
        int files = 0, finals = 0;
        foreach (string path in Directory.EnumerateFiles(after, "*.json").Order(StringComparer.Ordinal))
        {
            using var expected = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(before, Path.GetFileName(path))));
            using var actual = JsonDocument.Parse(File.ReadAllBytes(path));
            foreach (string key in new[] { "text", "partials" })
                if (expected.RootElement.GetProperty(key).GetRawText() != actual.RootElement.GetProperty(key).GetRawText()) throw new InvalidDataException("Live row mismatch: " + path + " " + key);
            var e = expected.RootElement.GetProperty("finals"); var a = actual.RootElement.GetProperty("finals");
            if (e.GetArrayLength() != a.GetArrayLength()) throw new InvalidDataException("Final count differs: " + path);
            for (int i = 0; i < e.GetArrayLength(); i++)
            {
                foreach (string key in new[] { "text", "start", "end", "available" })
                    if (e[i].GetProperty(key).GetRawText() != a[i].GetProperty(key).GetRawText()) throw new InvalidDataException("Final field differs: " + path + " " + key);
                if (!e[i].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).SequenceEqual(a[i].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()))) throw new InvalidDataException("Final IDs differ: " + path);
                finals++;
            }
            files++;
        }
        if (files == 0 || files != Directory.EnumerateFiles(before, "*.json").Count()) throw new InvalidDataException("Live row file sets differ or are empty.");
        Console.WriteLine($"{files} recordings, {finals} final ID arrays and all endpoint fields identical to baseline.");
    }

    internal static void Batch(string directory, string split, string output, int begin, int count, int chunk)
    {
        // At most one decode/final can occur per push below the 250 ms speech minimum.
        if (chunk is < 1 or > 1600) throw new ArgumentOutOfRangeException(nameof(chunk), "Token-traced probes support pushes up to 100 ms.");
        using var recognizer = MoonshineRecognizer.Load(directory, new MoonshineOptions { Threads = 2 });
        using var vad = new SileroVad();
        Directory.CreateDirectory(output);
        foreach (string file in Directory.EnumerateFiles(split, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Skip(begin).Take(count))
        {
            string id = Path.GetFileNameWithoutExtension(file), path = Path.Combine(output, id + ".json");
            if (File.Exists(path)) continue;
            var clip = OneAudio.Audio.Load(file, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            using var session = recognizer.CreateLiveSession(vad, "en");
            var finals = new List<FinalEvent>();
            var finalIds = new List<int[]>();
            int[] latestIds = [];
            recognizer.TokenTrace = value => latestIds = value;
            int partials = 0;
            for (int start = 0; start < clip.Samples.Length; start += chunk)
                foreach (var update in session.Push(new AudioBuffer(clip.Samples.Slice(start, Math.Min(chunk, clip.Samples.Length - start)), 16000)))
                { if (update is FinalEvent final) { finals.Add(final); finalIds.Add(latestIds); } if (update is PartialEvent) partials++; }
            foreach (var update in session.Complete()) if (update is FinalEvent final) { finals.Add(final); finalIds.Add(latestIds); }
            using (var stream = File.Create(path + ".part")) using (var json = new Utf8JsonWriter(stream))
            {
                json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", string.Join(" ", finals.Select(f => f.Segment.Text))); json.WriteNumber("chunkSamples", chunk); json.WriteNumber("partials", partials); json.WriteStartArray("finals");
                for (int index = 0; index < finals.Count; index++)
                {
                    var final = finals[index];
                    json.WriteStartObject(); json.WriteStartArray("ids"); foreach (int idValue in finalIds[index]) json.WriteNumberValue(idValue); json.WriteEndArray(); json.WriteString("text", final.Segment.Text); json.WriteNumber("available", final.AudioTime.TotalSeconds);
                    if (final.Segment.Start is { } start) json.WriteNumber("start", start.TotalSeconds);
                    if (final.Segment.End is { } end) json.WriteNumber("end", end.TotalSeconds);
                    json.WriteEndObject();
                }
                json.WriteEndArray(); json.WriteEndObject();
            }
            File.Move(path + ".part", path); Console.WriteLine(id);
        }
    }
}
