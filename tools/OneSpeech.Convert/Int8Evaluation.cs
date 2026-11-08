using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Whisper;

internal static class Int8Evaluation
{
    internal static void Batch(string model, string split, string output, int begin, int count)
    {
        Directory.CreateDirectory(output);
        using var recognizer = WhisperRecognizer.Load(model, new WhisperOptions { Threads = 8, Temperatures = [0], WordTimestamps = false });
        foreach (string file in Directory.EnumerateFiles(split, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Skip(begin).Take(count))
        {
            string id = Path.GetFileNameWithoutExtension(file), path = Path.Combine(output, id + ".json");
            if (File.Exists(path)) continue;
            var clip = OneAudio.Audio.Load(file, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var result = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
            using (var stream = File.Create(path + ".part")) using (var json = new Utf8JsonWriter(stream))
            {
                json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", result.Text); json.WriteString("model", result.Models[0].Id); json.WriteEndObject();
            }
            File.Move(path + ".part", path);
            Console.WriteLine(id);
        }
    }
    internal static void Metrics(string model, string clipPath, string output, int threads)
    {
        var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        var samples = new float[60 * 16000];
        for (int i = 0; i < samples.Length; i++) samples[i] = clip.Samples.Span[i % clip.Samples.Length];
        var clock = Stopwatch.StartNew();
        using var recognizer = WhisperRecognizer.Load(model, new WhisperOptions { Threads = threads, Temperatures = [0], WordTimestamps = false });
        double loadSeconds = clock.Elapsed.TotalSeconds;
        _ = recognizer.Transcribe(new AudioBuffer(samples.AsMemory(0, 16000), 16000), "en", CancellationToken.None);
        clock.Restart();
        var transcript = recognizer.Transcribe(new AudioBuffer(samples, 16000), "en", CancellationToken.None);
        double seconds = clock.Elapsed.TotalSeconds;
        using var stream = File.Create(output); using var json = new Utf8JsonWriter(stream);
        json.WriteStartObject(); json.WriteString("status", "indicative"); json.WriteString("model", transcript.Models[0].Id); json.WriteNumber("threads", threads);
        json.WriteNumber("audioSeconds", 60); json.WriteNumber("seconds", seconds); json.WriteNumber("rtf", seconds / 60); json.WriteNumber("loadSeconds", loadSeconds);
        json.WriteNumber("peakRssBytes", Process.GetCurrentProcess().PeakWorkingSet64); json.WriteNumber("convertedBytes", OneSpeech.Catalog.ModelCatalog.Default.Get(transcript.Models[0].Id).ConvertedFiles.Sum(f => f.Size));
        json.WriteString("runtime", Environment.Version.ToString()); json.WriteEndObject();
        Console.WriteLine(seconds.ToString(CultureInfo.InvariantCulture));
    }
}
