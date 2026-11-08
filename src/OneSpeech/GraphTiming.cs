using System.Diagnostics;
using System.Threading;

namespace OneSpeech;

// Opt-in instrumentation for structural measurements only. Conditional call sites,
// including their arguments, are absent from ordinary builds.
internal static class GraphTiming
{
#if ONESPEECH_GRAPH_TIMING
    internal static bool IsCompiled => true;
#else
    internal static bool IsCompiled => false;
#endif
    private static readonly long[] ticks = new long[8];
    private static readonly long[] calls = new long[8];
    private static readonly long[] bytes = new long[8];
    [ThreadStatic] private static long started;
    [ThreadStatic] private static long allocated;
    [ThreadStatic] private static int category;

    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void Begin(int kind)
    {
        category = kind;
        allocated = GC.GetAllocatedBytesForCurrentThread();
        started = Stopwatch.GetTimestamp();
    }

    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void BeginOptional(object? graph, int kind)
    {
        if (graph is not null) Begin(kind);
    }

    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void EndOptional(object? graph)
    {
        if (graph is not null) End();
    }

    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void End()
    {
        long elapsed = Stopwatch.GetTimestamp() - started;
        long size = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Interlocked.Add(ref ticks[category], elapsed);
        Interlocked.Add(ref bytes[category], size);
        Interlocked.Increment(ref calls[category]);
    }

    internal static (long Ticks, long Calls, long Bytes)[] Snapshot() =>
        Enumerable.Range(0, ticks.Length).Select(i => (Interlocked.Read(ref ticks[i]), Interlocked.Read(ref calls[i]), Interlocked.Read(ref bytes[i]))).ToArray();

    private static readonly long[] phaseTicks = new long[4];
    private static long phaseStart;
    private static long phaseGraphs;

    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void BeginPhase()
    {
        phaseGraphs = ticks.Sum();
        phaseStart = Stopwatch.GetTimestamp();
    }
    internal static Action<int>? PhaseCompleted { get; set; }
    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void EndPhase(int phase)
    {
        Interlocked.Add(ref phaseTicks[phase], Stopwatch.GetTimestamp() - phaseStart - (ticks.Sum() - phaseGraphs));
        PhaseCompleted?.Invoke(phase);
    }
    internal static long[] PhaseSnapshot() => (long[])phaseTicks.Clone();

    // Exclusive coarse costs: model loading, features, CTC postprocessing, clustering, audio I/O.
    private static readonly long[] costTicks = new long[5];
    [ThreadStatic] private static long[]? costStarted;
    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void BeginCost(int cost)
    {
        costStarted ??= new long[5];
        costStarted[cost] = Stopwatch.GetTimestamp();
    }
    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void EndCost(int cost)
    {
        long start = costStarted?[cost] ?? 0;
        if (start == 0) throw new InvalidOperationException("A measured cost ended without a matching start.");
        long elapsed = Stopwatch.GetTimestamp() - start;
        costStarted![cost] = 0;
        Interlocked.Add(ref costTicks[cost], elapsed);
    }
    internal static long[] CostSnapshot() => (long[])costTicks.Clone();

    internal static int ThreadCount { get; set; }
    [Conditional("ONESPEECH_GRAPH_TIMING")]
    internal static void OverrideThreads(ref int threads)
    {
        if (ThreadCount > 0) threads = ThreadCount;
    }

    internal static long TotalBytes => bytes.Sum();

    internal static void Reset()
    {
        if (costStarted is not null) Array.Clear(costStarted);
        Array.Clear(costTicks); Array.Clear(phaseTicks); Array.Clear(ticks); Array.Clear(calls); Array.Clear(bytes);
    }
}
