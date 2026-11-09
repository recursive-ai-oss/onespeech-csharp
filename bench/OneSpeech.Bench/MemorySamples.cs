using System.Diagnostics;
using System.Text.Json;
using System.Globalization;
using RecursiveAI.OneModel.Tensors;

internal sealed class MemorySamples : IDisposable
{
    private readonly StreamWriter writer;
    private readonly Timer timer;
    private readonly Stopwatch clock = Stopwatch.StartNew();
    private readonly Process process = Process.GetCurrentProcess();
    private readonly Func<string> phase;
    private readonly List<(double Seconds, long Bytes, string Phase)> samples = [];
    internal long Peak { get; private set; }
    internal long ProcessPeak { get; private set; }
    internal long SteadyMin { get; private set; } = long.MaxValue;
    internal long SteadyMax { get; private set; }
    internal List<string> Loads { get; } = [];
    internal MemorySamples(string file, Func<string> phase)
    {
        this.phase = phase;
        writer = new StreamWriter(file) { AutoFlush = true };
        writer.WriteLine("elapsed_s,working_set_bytes,phase,load,managed_heap_bytes,pool_idle_bytes,pool_outstanding_bytes,gen2_collections,managed_committed_bytes");
        Sample();
        timer = new Timer(_ => { lock (writer) Sample(); }, null, 5000, 5000);
    }
    private readonly Dictionary<string, long> phaseEnds = new(StringComparer.Ordinal);
    internal void CaptureBoundary(string stage)
    {
        lock (writer)
        {
            Sample(stage);
            phaseEnds[stage] = samples[^1].Bytes;
        }
    }
    internal void WritePhaseEnds(Utf8JsonWriter json)
    {
        json.WriteStartObject("phaseEndWorkingSetBytes");
        foreach (var entry in phaseEnds) json.WriteNumber(entry.Key, entry.Value);
        json.WriteEndObject();
    }
    private void Sample(string? overridePhase = null)
    {
        process.Refresh(); long bytes = process.WorkingSet64;
        double seconds = clock.Elapsed.TotalSeconds;
        string stage = overridePhase ?? phase();
        samples.Add((seconds, bytes, stage));
        Peak = Math.Max(Peak, bytes);
        if (seconds >= 30 && stage != "complete") { SteadyMin = Math.Min(SteadyMin, bytes); SteadyMax = Math.Max(SteadyMax, bytes); }
        string load = Machine.Load(); Loads.Add(load);
        var pool = BufferPool.Shared.GetStats();
        writer.WriteLine(FormattableString.Invariant($"{seconds:F3},{bytes},{stage},\"{load.Replace("\"", "\"\"", StringComparison.Ordinal)}\",{GC.GetTotalMemory(false)},{pool.IdleBytes},{pool.OutstandingBytes},{GC.CollectionCount(2)},{GC.GetGCMemoryInfo().TotalCommittedBytes}"));
    }
    internal void WriteSlopes(Utf8JsonWriter json) => WriteSlopes(json, samples);

    internal static void Summarize(string input, string output)
    {
        var rows = File.ReadLines(input).Skip(1).Select(line => line.Split(',', 4)).Select(fields =>
            (Seconds: double.Parse(fields[0], CultureInfo.InvariantCulture), Bytes: long.Parse(fields[1], CultureInfo.InvariantCulture), Phase: fields[2])).ToArray();
        using var stream = File.Create(output);
        using var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteBoolean("completed", rows.Length > 0 && rows[^1].Phase == "complete");
        WriteSlopes(json, rows);
        json.WriteEndObject();
    }

    private static void WriteSlopes(Utf8JsonWriter json, IReadOnlyList<(double Seconds, long Bytes, string Phase)> samples)
    {
        json.WriteStartArray("workingSetTrends");
        foreach (string stage in samples.Select(s => s.Phase).Where(p => p is not ("complete" or "idle")).Distinct(StringComparer.Ordinal).Prepend("all"))
        {
            var selected = samples.Where(s => stage == "all" ? s.Phase is not ("complete" or "idle") : s.Phase == stage).ToArray();
            // Each stage has distinct graph workspaces. Ignore its first minute to
            // separate initial shape-plan growth from a steady linear trend.
            var steady = selected.Length == 0 ? [] : selected.Where(s => s.Seconds >= selected[0].Seconds + 60).ToArray();
            json.WriteStartObject(); json.WriteString("phase", stage); json.WriteNumber("samples", steady.Length);
            if (steady.Length >= 3)
            {
                double slope = Slope(steady.Select(s => (s.Seconds, (double)s.Bytes)).ToArray());
                int width = Math.Max(1, steady.Length / 5);
                json.WriteNumber("slopeBytesPerSecond", slope);
                json.WriteNumber("firstFifthMeanBytes", steady.Take(width).Average(s => s.Bytes));
                json.WriteNumber("lastFifthMeanBytes", steady.TakeLast(width).Average(s => s.Bytes));
                json.WriteNumber("minBytes", steady.Min(s => s.Bytes)); json.WriteNumber("maxBytes", steady.Max(s => s.Bytes));
            }
            else json.WriteNull("slopeBytesPerSecond");
            json.WriteEndObject();
        }
        json.WriteEndArray();
    }
    internal static double Slope((double X, double Y)[] points)
    {
        if (points.Length < 2) throw new ArgumentException("At least two samples are required.", nameof(points));
        double mx = points.Average(p => p.X), my = points.Average(p => p.Y);
        double covariance = 0, variance = 0;
        foreach (var (x, y) in points) { covariance += (x - mx) * (y - my); variance += (x - mx) * (x - mx); }
        if (variance == 0) throw new ArgumentException("Sample times must vary.", nameof(points));
        return covariance / variance;
    }
    internal static void CheckSlope()
    {
        if (Slope([(0, 1024), (5, 1024), (10, 1024)]) != 0 || Slope([(0, 7), (5, 17), (10, 27)]) != 2)
            throw new InvalidOperationException("Working-set slope arithmetic failed.");
    }
    public void Dispose()
    {
        using var done = new ManualResetEvent(false);
        timer.Dispose(done); done.WaitOne(); Sample();
        ProcessPeak = process.PeakWorkingSet64;
        if (SteadyMin == long.MaxValue) SteadyMin = SteadyMax = Peak;
        writer.Dispose(); process.Dispose();
    }
}
