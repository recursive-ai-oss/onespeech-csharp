using System.Buffers.Binary;
using System.Diagnostics;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Vad;

if (args.Length != 2) throw new ArgumentException("Usage: AudioChecks parity|memory directory");
Directory.CreateDirectory(args[1]);
using var vad = new SileroVad();
if (args[0] == "parity")
{
    using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(args[1], "reference.json")));
    foreach (var probe in document.RootElement.GetProperty("probes").EnumerateArray())
    {
        int rate = probe.GetProperty("rate").GetInt32();
        var bytes = File.ReadAllBytes(Path.Combine(args[1], probe.GetProperty("file").GetString()!));
        var samples = new float[bytes.Length / 4];
        for (int i = 0; i < samples.Length; i++) samples[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        using var state = vad.CreateState(rate);
        var frame = new float[state.FrameSamples];
        var reference = probe.GetProperty("probabilities").EnumerateArray().Select(p => p.GetDouble()).ToArray();
        double max = 0, sum = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            frame.AsSpan().Clear();
            samples.AsSpan(i * frame.Length, Math.Min(frame.Length, samples.Length - i * frame.Length)).CopyTo(frame);
            double error = Math.Abs(state.ProcessFrame(frame, CancellationToken.None) - reference[i]);
            max = Math.Max(max, error); sum += error;
        }
        var intervals = vad.Detect(new AudioBuffer(samples, rate), CancellationToken.None);
        var expected = probe.GetProperty("intervals").EnumerateArray().ToArray();
        double boundary = 0, maximumBoundary = 0;
        if (intervals.Length != expected.Length) throw new InvalidOperationException($"Segment count {intervals.Length} != {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            var ds = Math.Abs(intervals[i].Start.TotalSeconds - expected[i].GetProperty("start").GetDouble() / rate) * 1000;
            var de = Math.Abs(intervals[i].End.TotalSeconds - expected[i].GetProperty("end").GetDouble() / rate) * 1000;
            boundary += ds + de;
            maximumBoundary = Math.Max(maximumBoundary, Math.Max(ds, de));
        }
        Console.WriteLine($"{probe.GetProperty("id").GetString()}: frames={reference.Length}, max={max:G9}, mean={sum / reference.Length:G9}, segments={intervals.Length}, meanBoundaryMs={(expected.Length == 0 ? 0 : boundary / (2 * expected.Length)):G9}, maxBoundaryMs={maximumBoundary:G9}");
        if (max > 1e-5 || maximumBoundary > 1000d / rate + 1e-6) throw new InvalidOperationException("Silero parity tolerance exceeded.");
    }
}
else if (args[0] == "memory")
{
    string path = Path.Combine(args[1], "two-hours.wav");
    const int seconds = 7200, rate = 16000;
    try
    {
        using (var file = File.Create(path))
        using (var writer = new BinaryWriter(file))
        {
            writer.Write("RIFF"u8); writer.Write(36 + seconds * rate * 2); writer.Write("WAVEfmt "u8);
            writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(rate); writer.Write(rate * 2);
            writer.Write((short)2); writer.Write((short)16); writer.Write("data"u8); writer.Write(seconds * rate * 2);
            file.SetLength(44L + seconds * rate * 2); // Deterministic digital silence; no voice or random seed.
        }
        // Warm the graph and allocation pools on the same horizon before sampling.
        vad.Detect(new AudioBuffer(new float[31 * rate], rate), CancellationToken.None);
        using var process = Process.GetCurrentProcess();
        process.Refresh(); long start = process.WorkingSet64;
        var progress = new MemoryProgress(process, start);
        using var source = AudioFileSource.Open(path, CancellationToken.None);
        int chunks = new VadChunker(vad).Read(Blocks(source), progress, CancellationToken.None).Count();
        process.Refresh();
        Console.WriteLine($"start={start}, peakSampled={progress.Peak}, peakOS={process.PeakWorkingSet64}, end={process.WorkingSet64}, chunks={chunks}, processed={progress.Time.TotalSeconds}s");
        if (progress.Time != TimeSpan.FromSeconds(seconds)) throw new InvalidOperationException("Audio length changed.");
    }
    finally { File.Delete(path); }
}
else throw new ArgumentException("Unknown mode");

static IEnumerable<AudioBuffer> Blocks(AudioFileSource source)
{
    var samples = new float[4096 * source.Channels];
    int frames;
    while ((frames = source.Read(samples, CancellationToken.None)) > 0)
        yield return new AudioBuffer(samples.AsMemory(0, frames * source.Channels), 16000, source.Channels);
}

sealed class MemoryProgress(Process process, long initial) : IProgress<TimeSpan>
{
    public long Peak { get; private set; } = initial;
    public TimeSpan Time { get; private set; }
    public void Report(TimeSpan value)
    {
        process.Refresh(); Peak = Math.Max(Peak, process.WorkingSet64); Time = value;
        if ((int)value.TotalSeconds % 300 < 30) Console.WriteLine($"audioSeconds={value.TotalSeconds:F0}, workingSet={process.WorkingSet64}, load={(File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : "unavailable")}");
    }
}
