using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Moonshine;
using OneSpeech.Vad;

if (args is ["--memory-summary", var memoryInput, var summaryOutput])
{
    MemorySamples.Summarize(memoryInput, summaryOutput);
    return 0;
}
if (args is ["--memory-file", var memoryFile, var memoryModel, var memoryThreads, var memoryOutput])
{
    if (!GraphTiming.IsCompiled) throw new InvalidOperationException("Build with -p:OneSpeechGraphTiming=true.");
    FileMemoryWorkload.Run(memoryFile, memoryModel, int.Parse(memoryThreads, CultureInfo.InvariantCulture), memoryOutput);
    return 0;
}
if (args is ["--self-test"]) { MemorySamples.CheckSlope(); Console.WriteLine("Slope checks passed; OneModel " + Machine.ModelPackageVersion); return 0; }
if (args is ["--prepare-data", var corpus, var meeting, var destination])
{
    Fixtures.Prepare(corpus, meeting, destination);
    return 0;
}
if (args is ["--link", var linkedModel, var existingDirectory])
{
    var entry = ModelCatalog.Default.Get(linkedModel);
    string key = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(entry.Id + "\n" + entry.Revision + "\n" + string.Join('\n', entry.Files.Select(f => f.Name + ":" + f.Sha256)))));
    Directory.CreateDirectory(ModelStore.Default.CacheDirectory);
    string link = Path.Combine(ModelStore.Default.CacheDirectory, key);
    if (!Directory.Exists(link)) Directory.CreateSymbolicLink(link, Path.GetFullPath(existingDirectory));
    Console.WriteLine(link);
    return 0;
}
if (args is ["--pull", var model])
{
    Console.WriteLine((await ModelStore.Default.PullAsync(model)).Directory);
    return 0;
}
if (args.Length != 8 || args[0] != "--filter" || args[2] != "--threads" || args[4] != "--data" || args[6] != "--out")
{
    Console.Error.WriteLine("Usage: --filter all|turbo60|turbo600|long|small|qwen|live|diarize|align --threads N --data DIR --out DIR\nPreparation: --prepare-data LIBRISPEECH_DIR AMI_WAV DIR; --pull MODEL_ID");
    return 2;
}
if (!GraphTiming.IsCompiled) throw new InvalidOperationException("Build with -p:OneSpeechGraphTiming=true. Ordinary builds deliberately contain no timing calls.");
int threads = int.Parse(args[3], CultureInfo.InvariantCulture);
if (threads is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(args), "Threads must be 1–256.");
GraphTiming.ThreadCount = threads;
string data = Path.GetFullPath(args[5]), output = Path.GetFullPath(args[7]);
Directory.CreateDirectory(output);
string[] names = ["turbo60", "turbo600", "long", "small", "qwen", "live", "diarize", "align"];
if (args[1] != "all" && !names.Contains(args[1], StringComparer.Ordinal)) throw new ArgumentException("Unknown filter.");
foreach (string name in names.Where(n => args[1] == "all" || args[1] == n))
{
    string file = Path.Combine(data, name switch { "turbo600" => "clip600.wav", "long" => "long.wav", "diarize" => "ami.wav", _ => "clip60.wav" });
    double duration = Fixtures.Duration(file);
    string id = name switch { "long" => "openai/whisper-tiny", "small" => "openai/whisper-small", "qwen" => "Qwen/Qwen3-ASR-1.7B", "live" => "moonshine-ai/moonshine-streaming-tiny", "align" => "facebook/omniASR-CTC-300M", "diarize" => "iic/speech_campplus_sv_en_voxceleb_16k", _ => "openai/whisper-large-v3-turbo" };
    var load = Stopwatch.StartNew();
    // Pull before either pass: network/conversion are never throughput measurements.
    var resolved = await ModelStore.Default.PullAsync(id);
    using var workload = name == "live" ? null : await LoadedWorkload.LoadAsync(name, id, threads);
    load.Stop();
    double modelLoadSeconds = workload?.LoadSeconds ?? 0;
    double prepareSeconds = load.Elapsed.TotalSeconds - modelLoadSeconds;
    var warmClock = Stopwatch.StartNew();
    if (workload is not null) _ = workload.Run(Path.Combine(data, "warm.wav"), File.ReadAllText(Path.Combine(data, "warm-align.txt")), Fixtures.Duration(Path.Combine(data, "warm.wav")));
    warmClock.Stop();
    double warmupSeconds = warmClock.Elapsed.TotalSeconds;
    var sampler = new MemorySamples(Path.Combine(output, $"{name}-{threads}.memory.csv"), () => workload?.Phase ?? "live");
    string beforeLoad = Machine.Load();
    GraphTiming.Reset();
    long allocated = GC.GetTotalAllocatedBytes(precise: true);
    long start = Stopwatch.GetTimestamp();
    long ourLiveBytes = 0, liveChunks = 0, nonemittingBytes = 0, nonemittingChunks = 0;
    double setupSeconds = 0;
    Transcript? document = null;
    if (name == "live")
    {
        long setup = Stopwatch.GetTimestamp();
        using var recognizer = MoonshineRecognizer.Load(resolved.Directory);
        using var vad = new SileroVad();
        modelLoadSeconds = Stopwatch.GetElapsedTime(setup).TotalSeconds;
        var audio = Fixtures.ReadMono(file);
        var warmAudio = Fixtures.ReadMono(Path.Combine(data, "warm.wav"));
        warmClock.Restart();
        using (var warm = recognizer.CreateLiveSession(vad, "en")) Push(warm, warmAudio, false);
        warmClock.Stop(); warmupSeconds = warmClock.Elapsed.TotalSeconds;
        using var session = recognizer.CreateLiveSession(vad, "en");
        setupSeconds = Stopwatch.GetElapsedTime(setup).TotalSeconds;
        GraphTiming.Reset();
        allocated = GC.GetTotalAllocatedBytes(precise: true);
        start = Stopwatch.GetTimestamp();
        Push(session, audio, true);
        void Push(LiveSession session, float[] audio, bool measure)
        {
            int index = 0;
            for (int offset = 0; offset < audio.Length; offset += 1600)
            {
                var block = new AudioBuffer(audio.AsMemory(offset, Math.Min(1600, audio.Length - offset)), 16000);
                long graphBefore = GraphTiming.TotalBytes, bytesBefore = GC.GetAllocatedBytesForCurrentThread();
                var events = session.Push(block);
                long bytes = GC.GetAllocatedBytesForCurrentThread() - bytesBefore;
                long modelBytes = GraphTiming.TotalBytes - graphBefore;
                if (measure && index >= 100)
                {
                    ourLiveBytes += bytes - modelBytes - Payload.Bytes(events); liveChunks++;
                    if (events.IsEmpty) { nonemittingBytes += bytes - modelBytes; nonemittingChunks++; }
                }
                index++;
            }
            _ = session.Complete();
        }
    }
    else document = workload!.Run(file, File.ReadAllText(Path.Combine(data, "align.txt")), duration);
    double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
    long bytesAllocated = GC.GetTotalAllocatedBytes(precise: true) - allocated;
    var graphs = GraphTiming.Snapshot();
    var phases = GraphTiming.PhaseSnapshot();
    var costTicks = GraphTiming.CostSnapshot();
    sampler.Dispose();
    GC.KeepAlive(document);
    double graphSeconds = graphs.Sum(x => x.Ticks) / (double)Stopwatch.Frequency;
    if (graphSeconds <= 0) throw new InvalidOperationException("No graph calls were recorded; the row is invalid.");
    if (costTicks.Sum() / (double)Stopwatch.Frequency > seconds - graphSeconds + 0.01)
        throw new InvalidOperationException("Exclusive profile costs overlap or escaped the measured interval.");
    string stem = Path.Combine(output, $"{name}-{threads}");
    using (var stream = File.Create(stem + ".json"))
    using (var json = new Utf8JsonWriter(stream, new() { Indented = true }))
    {
        json.WriteStartObject();
        json.WriteNumber("measurementVersion", 5);
        if (document is not null) json.WriteNumber("words", document.Words.Length);
        json.WriteString("kind", "indicative"); json.WriteString("row", name); json.WriteString("model", id);
        json.WriteString("revision", resolved.Info.Revision); json.WriteString("precision", "fp32"); json.WriteNumber("threads", threads);
        json.WriteString("inputSha256", Fixtures.Hash(file)); json.WriteNumber("audioSeconds", duration);
        json.WriteNumber("prepareSeconds", prepareSeconds); json.WriteNumber("modelLoadSeconds", modelLoadSeconds); json.WriteNumber("warmupSeconds", warmupSeconds); json.WriteNumber("liveSetupSeconds", setupSeconds);
        json.WriteString("warmupInputSha256", Fixtures.Hash(Path.Combine(data, "warm.wav")));
        sampler.WriteSlopes(json);
        json.WriteNumber("endToEndSeconds", seconds); json.WriteNumber("graphSeconds", graphSeconds);
        if (graphSeconds > 0) json.WriteNumber("ratio", seconds / graphSeconds); else json.WriteNull("ratio");
        json.WriteNumber("rtf", seconds / duration); json.WriteNumber("processAllocatedBytes", bytesAllocated);
        json.WriteNumber("liveWorkingBytes", ourLiveBytes); json.WriteNumber("liveMeasuredChunks", liveChunks);
        json.WriteNumber("liveNonemittingBytes", nonemittingBytes); json.WriteNumber("liveNonemittingChunks", nonemittingChunks);
        json.WriteNumber("sampledPeakWorkingSetBytes", sampler.Peak); json.WriteNumber("processLifetimePeakBytes", sampler.ProcessPeak); json.WriteNumber("memorySamplePeriodSeconds", 5); json.WriteNumber("steadyMinBytes", sampler.SteadyMin); json.WriteNumber("steadyMaxBytes", sampler.SteadyMax);
        json.WriteStartArray("graphs");
        string[] labels = ["omniASR", "embedder", "silero", "qwen", "whisper", "moonshine", "reserved6", "reserved7"];
        for (int i = 0; i < graphs.Length; i++) { json.WriteStartObject(); json.WriteString("category", labels[i]); json.WriteNumber("seconds", graphs[i].Ticks / (double)Stopwatch.Frequency); json.WriteNumber("calls", graphs[i].Calls); json.WriteNumber("allocatedBytes", graphs[i].Bytes); json.WriteEndObject(); }
        json.WriteEndArray();
        json.WriteStartObject("nonGraphPhaseSeconds");
        string[] costs = ["modelResolution", "recognition", "alignment", "diarization"];
        for (int i = 0; i < phases.Length; i++) json.WriteNumber(costs[i], phases[i] / (double)Stopwatch.Frequency);
        json.WriteNumber("other", seconds - graphSeconds - phases.Sum() / (double)Stopwatch.Frequency);
        json.WriteEndObject();
        json.WriteStartObject("exclusiveNonGraphSeconds");
        string[] costNames = ["modelLoading", "features", "ctcPostprocessing", "clustering", "audioIO"];
        for (int i = 0; i < costTicks.Length; i++) json.WriteNumber(costNames[i], costTicks[i] / (double)Stopwatch.Frequency);
        json.WriteNumber("unattributed", seconds - graphSeconds - costTicks.Sum() / (double)Stopwatch.Frequency);
        json.WriteEndObject();
        Machine.Write(json, beforeLoad, sampler.Loads);
        json.WriteEndObject();
    }
    string line = FormattableString.Invariant($"{name},{threads},{duration:F3},{modelLoadSeconds:F6},{seconds:F6},{graphSeconds:F6},{seconds / graphSeconds:F6},{seconds / duration:F6},{sampler.Peak},{sampler.SteadyMin},{sampler.SteadyMax},{ourLiveBytes},{liveChunks}");
    File.WriteAllText(stem + ".csv", "row,threads,audio_s,load_s,e2e_s,graph_s,ratio,rtf,peak_bytes,steady_min_bytes,steady_max_bytes,live_working_bytes,live_chunks\n" + line + "\n");
    File.WriteAllText(stem + ".md", "| Row | Threads | Load s | E2E s | Graph s | Ratio | RTF | Peak MB |\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + FormattableString.Invariant($"| {name} | {threads} | {modelLoadSeconds:F2} | {seconds:F2} | {graphSeconds:F2} | {seconds / graphSeconds:F3} | {seconds / duration:F3} | {sampler.Peak / 1e6:F0} |\n"));
    Console.WriteLine(line);
}
return 0;

internal static class Payload
{
    // Reproduce only the immutable payload graph, outside the measured push.
    internal static long Bytes(ImmutableArray<LiveEvent> events)
    {
        if (events.IsEmpty) return 0;
        long start = GC.GetAllocatedBytesForCurrentThread();
        var copy = new LiveEvent[events.Length];
        for (int i = 0; i < events.Length; i++) copy[i] = events[i] switch
        {
            PartialEvent p => p with { Segment = Copy(p.Segment) },
            FinalEvent f => f with { Segment = Copy(f.Segment) },
            EndpointEvent e => e with { },
            _ => throw new InvalidOperationException("Unmeasured live event shape.")
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

internal static class Machine
{
    internal static string ModelPackageVersion => typeof(RecursiveAI.OneModel.Graph.Session).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "unknown";
    internal static string Command(string command, string argument)
    {
        using var process = Process.Start(new ProcessStartInfo(command, argument) { RedirectStandardOutput = true, UseShellExecute = false });
        if (process is null) return "unavailable";
        string result = process.StandardOutput.ReadToEnd().Trim(); process.WaitForExit(); return result;
    }
    internal static string Load() => OperatingSystem.IsLinux() ? File.ReadAllText("/proc/loadavg").Trim() : Command("sysctl", "-n vm.loadavg");
    internal static void Write(Utf8JsonWriter json, string before, List<string> during)
    {
        json.WriteStartObject("machine");
        json.WriteString("host", Environment.MachineName); json.WriteString("os", RuntimeInformation.OSDescription);
        json.WriteString("architecture", RuntimeInformation.ProcessArchitecture.ToString()); json.WriteString("runtime", RuntimeInformation.FrameworkDescription);
        json.WriteString("sdk", Command("dotnet", "--version"));
        json.WriteString("cpu", OperatingSystem.IsLinux() ? File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal)) : Command("sysctl", "-n machdep.cpu.brand_string"));
        json.WriteNumber("processCores", Environment.ProcessorCount);
        json.WriteString("onlineCores", OperatingSystem.IsLinux() ? Command("getconf", "_NPROCESSORS_ONLN") : Command("sysctl", "-n hw.logicalcpu"));
        json.WriteString("oneModelPackage", ModelPackageVersion);
        json.WriteString("isa", $"AVX2={Avx2.IsSupported};AVX512F={Avx512F.IsSupported};FMA={Fma.IsSupported};AdvSimd={AdvSimd.IsSupported}");
        json.WriteString("oneModel", typeof(RecursiveAI.OneModel.Graph.Session).Assembly.GetName().Version?.ToString());
        json.WriteString("oneAudio", typeof(OneAudio.Audio).Assembly.GetName().Version?.ToString());
        json.WriteString("loadBefore", before); json.WriteStartArray("loadDuring"); foreach (string load in during) json.WriteStringValue(load); json.WriteEndArray(); json.WriteString("loadAfter", Load());
        json.WriteString("gpu", "unused"); json.WriteEndObject();
    }
}
