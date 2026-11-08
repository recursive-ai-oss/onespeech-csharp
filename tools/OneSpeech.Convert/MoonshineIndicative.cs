using System.Diagnostics;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Moonshine;
using OneSpeech.Vad;

internal static class MoonshineIndicative
{
    internal static void Run(string directory, string path, string output)
    {
        var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        var load = Stopwatch.StartNew();
        using var model = MoonshineRecognizer.Load(directory, new MoonshineOptions { Threads = 1 });
        using var vad = new SileroVad();
        load.Stop();
        Measure(); // Warm JIT, kernels and tensor pools before the recorded repetitions.
        using var file = File.Create(output); using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject(); json.WriteString("status", "indicative, not a publishable benchmark"); json.WriteString("workload", "committed 1089-134686-0001.flac, mono16kHz, 100ms pushes, 200ms partial cadence, Silero endpointing"); json.WriteString("precision", "fp32"); json.WriteNumber("inferenceThreads", 1); json.WriteNumber("processorCount", Environment.ProcessorCount); json.WriteString("runtime", Environment.Version.ToString()); json.WriteString("oneModel", "1.2.0"); json.WriteString("oneAudio", "1.7.2"); json.WriteBoolean("avx2", System.Runtime.Intrinsics.X86.Avx2.IsSupported); json.WriteBoolean("fma", System.Runtime.Intrinsics.X86.Fma.IsSupported); json.WriteBoolean("avx512f", System.Runtime.Intrinsics.X86.Avx512F.IsSupported); json.WriteNumber("modelAndVadLoadSeconds", load.Elapsed.TotalSeconds); json.WriteStartArray("repetitions");
        for (int i = 0; i < 3; i++)
        {
            var row = Measure();
            json.WriteStartObject(); json.WriteNumber("elapsedSeconds", row.Seconds); json.WriteNumber("audioSeconds", clip.Samples.Length / 16000d); json.WriteNumber("realTimeFactor", row.Seconds / (clip.Samples.Length / 16000d)); json.WriteNumber("steadyChunks", row.Chunks); json.WriteNumber("steadyAllocatedBytes", row.Bytes); json.WriteNumber("steadyBytesPerChunk", row.Bytes / (double)row.Chunks); json.WriteNumber("minimumChunkAllocatedBytes", row.Minimum); json.WriteNumber("completionAllocatedBytes", row.CompleteBytes); json.WriteNumber("partials", row.Partials); json.WriteNumber("finals", row.Finals); json.WriteEndObject();
        }
        json.WriteEndArray(); json.WriteEndObject();
        (double Seconds, int Chunks, long Bytes, long Minimum, long CompleteBytes, int Partials, int Finals) Measure()
        {
            using var session = model.CreateLiveSession(vad, "en");
            int partials = 0, finals = 0, chunks = 0; long allocated = 0, minimum = long.MaxValue;
            var timer = Stopwatch.StartNew();
            for (int start = 0; start < clip.Samples.Length; start += 1600)
            {
                var input = new AudioBuffer(clip.Samples.Slice(start, Math.Min(1600, clip.Samples.Length - start)), 16000);
                long before = GC.GetTotalAllocatedBytes(precise: true);
                var events = session.Push(input);
                long bytes = GC.GetTotalAllocatedBytes(precise: true) - before;
                if (start > 0) { chunks++; allocated += bytes; minimum = Math.Min(minimum, bytes); }
                foreach (var update in events) { if (update is PartialEvent) partials++; if (update is FinalEvent) finals++; }
            }
            long finish = GC.GetTotalAllocatedBytes(precise: true);
            foreach (var update in session.Complete()) { if (update is PartialEvent) partials++; if (update is FinalEvent) finals++; }
            long completeBytes = GC.GetTotalAllocatedBytes(precise: true) - finish;
            timer.Stop();
            return (timer.Elapsed.TotalSeconds, chunks, allocated, minimum, completeBytes, partials, finals);
        }
    }
}
