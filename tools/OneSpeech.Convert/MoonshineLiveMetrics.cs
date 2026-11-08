using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Live;
using OneSpeech.Moonshine;
using OneSpeech.Vad;

internal static class MoonshineLiveMetrics
{
    internal static void Inspect(string directory, string output)
    {
        using var model = MoonshineRecognizer.Load(directory, new MoonshineOptions { Threads = 1 });
        using var workspace = new MoonshineLiveWorkspace(model);
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file);
        json.WriteStartObject();
        workspace.InspectRuns((name, action) =>
        {
            for (int i = 0; i < 20; i++) action();
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++) action();
            json.WriteNumber(name, (GC.GetAllocatedBytesForCurrentThread() - before) / 32d);
        });
        json.WriteEndObject();
    }
    internal static void Run(string directory, string clipPath, string output, int threads = 1, bool sixtySeconds = false)
    {
        var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        ReadOnlyMemory<float> pcm = clip.Samples;
        if (sixtySeconds)
        {
            var samples = new float[60 * 16000];
            for (int i = 0; i < samples.Length; i++) samples[i] = clip.Samples.Span[i % clip.Samples.Length];
            pcm = samples;
        }
        using var model = MoonshineRecognizer.Load(directory, new MoonshineOptions { Threads = threads });
        bool streaming = MoonshineConfiguration.Load(directory).Streaming;
        using var vad = new SileroVad();
        Measure();
        // Time the public session path separately from allocation instrumentation.
        double PublicSeconds()
        {
            using var live = model.CreateLiveSession(vad, "en");
            var timer = Stopwatch.StartNew();
            for (int start = 0; start < pcm.Length; start += 1600)
                _ = live.Push(new AudioBuffer(pcm.Slice(start, Math.Min(1600, pcm.Length - start)), 16000));
            _ = live.Complete();
            return timer.Elapsed.TotalSeconds;
        }
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("status", "indicative; immutable emitted payload excluded; actual VAD runtime allocation measured separately");
        json.WriteString("runtime", Environment.Version.ToString()); json.WriteString("oneModel", "1.2.0-preview.2"); json.WriteString("oneAudio", "1.7.2"); json.WriteNumber("processorCount", Environment.ProcessorCount);
        json.WriteNumber("inferenceThreads", threads); json.WriteBoolean("streaming", streaming); json.WriteNumber("convertedBytes", Directory.EnumerateFiles(directory, "*.onemodel").Sum(p => new FileInfo(p).Length));
        json.WriteStartArray("repetitions");
        for (int i = 0; i < (sixtySeconds ? 1 : 3); i++)
        {
            double publicSeconds = PublicSeconds();
            var row = Measure(); json.WriteStartObject(); json.WriteNumber("publicRtf", publicSeconds / (pcm.Length / 16000d)); json.WriteNumber("rtf", row.Seconds / (pcm.Length / 16000d));
            json.WriteNumber("audioSeconds", pcm.Length / 16000d); json.WriteNumber("seconds", row.Seconds); json.WriteNumber("chunks", row.Chunks);
            json.WriteNumber("totalBytes", row.Total); json.WriteNumber("oneModelVadBytes", row.Vad); json.WriteNumber("emittedPayloadBytes", row.Payload); json.WriteNumber("ourWorkingBytes", row.Total - row.Vad - row.Payload);
            json.WriteNumber("nonEmittingChunks", row.Quiet); json.WriteNumber("nonEmittingOurBytes", row.QuietBytes); json.WriteNumber("partials", row.Partials); json.WriteNumber("finals", row.Finals); json.WriteEndObject();
        }
        json.WriteEndArray(); json.WriteNumber("peakRssBytes", Process.GetCurrentProcess().PeakWorkingSet64); json.WriteEndObject();
        (double Seconds, int Chunks, long Total, long Vad, long Payload, int Quiet, long QuietBytes, int Partials, int Finals) Measure()
        {
            long vadBytes = 0;
            using var state = vad.CreateState(16000);
            float Probability(ReadOnlyMemory<float> samples, CancellationToken ct)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                float p = state.ProcessFrame(samples.Span, ct);
                vadBytes += GC.GetAllocatedBytesForCurrentThread() - before;
                return p;
            }
            using var session = streaming
                ? new UtteranceLiveSession(model, Probability, new MoonshineStreamingState(model), "en")
                : new UtteranceLiveSession(model, Probability, "en");
            double seconds = 0; long total = 0, payload = 0, quietBytes = 0; int chunks = 0, quiet = 0, partials = 0, finals = 0;
            for (int start = 0; start < pcm.Length; start += 1600)
            {
                var input = new AudioBuffer(pcm.Slice(start, Math.Min(1600, pcm.Length - start)), 16000);
                long priorVad = vadBytes, before = GC.GetAllocatedBytesForCurrentThread(), tick = Stopwatch.GetTimestamp();
                var events = session.Push(input);
                seconds += Stopwatch.GetElapsedTime(tick).TotalSeconds;
                long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
                total += bytes; chunks++;
                long clone = Payload(events); payload += clone;
                if (events.IsEmpty) { quiet++; quietBytes += bytes - (vadBytes - priorVad); }
                foreach (var e in events) { if (e is PartialEvent) partials++; if (e is FinalEvent) finals++; }
            }
            long finish = GC.GetAllocatedBytesForCurrentThread(), finishTick = Stopwatch.GetTimestamp();
            var last = session.Complete(); seconds += Stopwatch.GetElapsedTime(finishTick).TotalSeconds;
            total += GC.GetAllocatedBytesForCurrentThread() - finish; payload += Payload(last);
            foreach (var e in last) if (e is FinalEvent) finals++;
            return (seconds, chunks, total, vadBytes, payload, quiet, quietBytes, partials, finals);
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
