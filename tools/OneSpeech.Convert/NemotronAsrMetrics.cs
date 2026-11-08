using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Live;
using OneSpeech.FastConformer;
using OneSpeech.Vad;

internal static class NemotronAsrMetrics
{
    internal static void Run(string directory, string clipPath, string output, int chunk, int threads = 1)
    {
        string Load() => OperatingSystem.IsLinux() ? File.ReadAllText("/proc/loadavg").Trim() : "unavailable";
        string loadBefore = Load(); long loadTick = Stopwatch.GetTimestamp();
        var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        using var model = NemotronAsrRecognizer.Load(new(directory, NemotronAsrEvaluation.Info()), new NemotronAsrOptions { Threads = threads, ChunkMilliseconds = chunk });
        double modelLoadSeconds = Stopwatch.GetElapsedTime(loadTick).TotalSeconds;
        using var vad = new SileroVad();
        Measure();
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("status", "indicative; immutable emitted payload excluded; actual VAD runtime allocation measured separately");
        json.WriteString("runtime", Environment.Version.ToString()); json.WriteString("oneModel", "1.2.0"); json.WriteString("oneAudio", "1.7.2"); json.WriteNumber("processorCount", Environment.ProcessorCount); json.WriteNumber("inferenceThreads", threads);
        json.WriteString("sdk", "10.0.403"); json.WriteString("loadBefore", loadBefore); json.WriteString("architecture", RuntimeInformation.ProcessArchitecture.ToString());
        json.WriteNumber("modelLoadSeconds", modelLoadSeconds); json.WriteString("modelLoadMethod", "warmed filesystem cache; includes clip decode and model construction, excludes VAD/session startup");
        json.WriteNumber("convertedBytes", OneSpeech.Catalog.ModelCatalog.Default.Get(NemotronAsrRecognizer.Identifier).ConvertedFiles.Sum(f => f.Size));
        if (OperatingSystem.IsLinux())
        {
            string[] cpu = File.ReadAllLines("/proc/cpuinfo");
            json.WriteString("cpu", cpu.First(line => line.StartsWith("model name", StringComparison.Ordinal)));
            json.WriteString("isa", cpu.First(line => line.StartsWith("flags", StringComparison.Ordinal)));
        }
        json.WriteStartArray("repetitions");
        for (int i = 0; i < 3; i++)
        {
            var row = Measure(); json.WriteStartObject(); json.WriteNumber("rtf", row.Seconds / (clip.Samples.Length / 16000d));
            json.WriteString("loadDuring", row.Load); json.WriteNumber("sessionStartupSeconds", row.Startup);
            json.WriteNumber("workingSetDuring", row.WorkingSet);
            json.WriteNumber("audioSeconds", clip.Samples.Length / 16000d); json.WriteNumber("seconds", row.Seconds); json.WriteNumber("chunks", row.Chunks);
            json.WriteNumber("totalBytes", row.Total); json.WriteNumber("oneModelVadBytes", row.Vad); json.WriteNumber("emittedPayloadBytes", row.Payload); json.WriteNumber("ourWorkingBytes", row.Total - row.Vad - row.Payload);
            json.WriteNumber("nonEmittingChunks", row.Quiet); json.WriteNumber("nonEmittingOurBytes", row.QuietBytes); json.WriteNumber("partials", row.Partials); json.WriteNumber("finals", row.Finals); json.WriteEndObject();
        }
        json.WriteEndArray(); json.WriteString("loadAfter", Load()); json.WriteEndObject();
        (double Seconds, int Chunks, long Total, long Vad, long Payload, int Quiet, long QuietBytes, int Partials, int Finals, string Load, double Startup, long WorkingSet) Measure()
        {
            long vadBytes = 0; long startupTick = Stopwatch.GetTimestamp();
            using var state = vad.CreateState(16000);
            using var session = new UtteranceLiveSession(model, (samples, ct) =>
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                float p = state.ProcessFrame(samples.Span, ct);
                vadBytes += GC.GetAllocatedBytesForCurrentThread() - before;
                return p;
            }, new NemotronAsrState(model, 0, CancellationToken.None), "en");
            double startup = Stopwatch.GetElapsedTime(startupTick).TotalSeconds; string loadDuring = "unavailable"; long workingSet = 0;
            double seconds = 0; long total = 0, payload = 0, quietBytes = 0; int chunks = 0, quiet = 0, partials = 0, finals = 0;
            for (int start = 0; start < clip.Samples.Length; start += 1600)
            {
                var input = new AudioBuffer(clip.Samples.Slice(start, Math.Min(1600, clip.Samples.Length - start)), 16000);
                long priorVad = vadBytes, before = GC.GetAllocatedBytesForCurrentThread(), tick = Stopwatch.GetTimestamp();
                var events = session.Push(input);
                seconds += Stopwatch.GetElapsedTime(tick).TotalSeconds;
                long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
                total += bytes; chunks++;
                if (chunks == Math.Max(1, clip.Samples.Length / 1600 / 2)) { loadDuring = Load(); workingSet = Process.GetCurrentProcess().WorkingSet64; }
                long clone = Payload(events); payload += clone;
                if (events.IsEmpty) { quiet++; quietBytes += bytes - (vadBytes - priorVad); }
                foreach (var e in events) { if (e is PartialEvent) partials++; if (e is FinalEvent) finals++; }
            }
            long finish = GC.GetAllocatedBytesForCurrentThread(), finishTick = Stopwatch.GetTimestamp();
            var last = session.Complete(); seconds += Stopwatch.GetElapsedTime(finishTick).TotalSeconds;
            total += GC.GetAllocatedBytesForCurrentThread() - finish; payload += Payload(last);
            foreach (var e in last) if (e is FinalEvent) finals++;
            return (seconds, chunks, total, vadBytes, payload, quiet, quietBytes, partials, finals, loadDuring, startup, workingSet);
        }
    }

    // Replay only the owned payload graph. No builders or temporary containers are measured.
    private static long Payload(ImmutableArray<LiveEvent> events)
    {
        if (events.IsEmpty) return 0;
        long start = GC.GetAllocatedBytesForCurrentThread();
        var copy = new LiveEvent[events.Length];
        for (int i = 0; i < events.Length; i++) copy[i] = events[i] switch
        {
            PartialEvent p => p with { Segment = Copy(p.Segment) },
            FinalEvent f => f with { Segment = Copy(f.Segment) },
            EndpointEvent e => e with { },
            _ => throw new InvalidOperationException("Unexpected live metric event.")
        };
        long bytes = GC.GetAllocatedBytesForCurrentThread() - start;
        GC.KeepAlive(copy); return bytes;
    }
    private static Segment Copy(Segment segment)
    {
        Word[] words = segment.Words.IsEmpty ? [] : new Word[segment.Words.Length];
        for (int i = 0; i < words.Length; i++) words[i] = segment.Words[i] with { Text = new string(segment.Words[i].Text.AsSpan()) };
        return segment with { Text = new string(segment.Text.AsSpan()), Words = words.Length == 0 ? [] : ImmutableCollectionsMarshal.AsImmutableArray(words) };
    }
}
