using System.Globalization;
using System.Collections.Immutable;
using System.Text;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Media;
using OneSpeech.Speakers;
using OneSpeech.Vad;

return await CommandLine.RunAsync(args, Console.Out, Console.Error);

internal static class CommandLine
{
    internal const string Help = """
        Speech recognition for .NET.
        Usage: onespeech <command> [arguments] [options]
          transcribe <audio>                 Full timed, diarized transcript
          align <audio> <text-file>           Align UTF-8 written text (up to 60 s)
          diarize <audio>                    Speaker turns only (no recognized words)
          enroll <name> <enrollment-audio>    Enroll for this process, then discard
          identify <audio> <name> <enrollment-audio> [<name> <enrollment-audio> ...]
          verify <first-audio> <second-audio>
          detect-language <audio>
          models list
          models pull <id>
        Options:
          --model <catalog-id>   Tier-2 models require an explicit name; gated models
                                use ONESPEECH_MODEL_TOKEN after gate acceptance.
          --language <tag>       Transcribe/align; en or zh embedder for speaker commands
          --format text|json|srt|vtt  Transcript output (default text)
          --output <file>        Write UTF-8 output instead of stdout
          --threshold <cosine>   Identify threshold in [-1,1] (default 0.3; not calibrated)
          --help                Show this help
          --version             Show version
        Enrollments and embeddings are in-memory only and never persisted. Identify
        takes enrollment audio in the same invocation. Obtain informed consent and
        check local law before processing voices (including BIPA and GDPR Art. 9).
        Exit codes: 0 success, 1 model/audio/I/O failure, 2 usage, 130 cancellation.
        Unaffiliated with other products or companies named OneSpeech.
        """;

    internal static async Task<int> RunAsync(string[] args, TextWriter output, TextWriter error)
    {
        using var cancellation = new CancellationTokenSource();
        ConsoleCancelEventHandler handler = (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        Console.CancelKeyPress += handler;
        try
        {
            int terminator = Array.IndexOf(args, "--");
            if (args.Length == 0 || args.Take(terminator < 0 ? args.Length : terminator).Contains("--help", StringComparer.Ordinal)) { await output.WriteLineAsync(Help); return 0; }
            if (args is ["--version"]) { await output.WriteLineAsync(BuildVersion.Value); return 0; }
            var options = Parse(args);
            MediaInput.Register();
            string result = await ExecuteAsync(options, cancellation.Token);
            if (options.Output is null) await output.WriteLineAsync(result);
            else await File.WriteAllTextAsync(options.Output, result, new UTF8Encoding(false), cancellation.Token);
            return 0;
        }
        catch (UsageException e) { await error.WriteLineAsync(e.Message + " Use --help for usage."); return 2; }
        catch (OneSpeechException e) { await error.WriteLineAsync($"{ErrorName(e)}: {e.Message}"); return 1; }
        catch (OperationCanceledException) { await error.WriteLineAsync("Cancelled."); return 130; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException)
        { await error.WriteLineAsync($"{ErrorName(e)}: {e.Message}"); return 1; }
        finally { Console.CancelKeyPress -= handler; }
    }

    private static string ErrorName(Exception error) => error switch
    {
        ModelNotFoundException => nameof(ModelNotFoundException),
        ModelIntegrityException => nameof(ModelIntegrityException),
        ModelTokenRequiredException => nameof(ModelTokenRequiredException),
        ModelTokenRejectedException => nameof(ModelTokenRejectedException),
        ModelGateNotAcceptedException => nameof(ModelGateNotAcceptedException),
        ModelNetworkException => nameof(ModelNetworkException),
        InvalidModelCatalogException => nameof(InvalidModelCatalogException),
        OneSpeech.Custom.InvalidModelShapeException => nameof(OneSpeech.Custom.InvalidModelShapeException),
        CapabilityNotSupportedException => nameof(CapabilityNotSupportedException),
        UnsupportedAudioException => nameof(UnsupportedAudioException),
        InvalidTranscriptJsonException => nameof(InvalidTranscriptJsonException),
        FeatureUnavailableException => nameof(FeatureUnavailableException),
        OneSpeechException => nameof(OneSpeechException),
        FileNotFoundException => nameof(FileNotFoundException),
        DirectoryNotFoundException => nameof(DirectoryNotFoundException),
        IOException => nameof(IOException),
        UnauthorizedAccessException => nameof(UnauthorizedAccessException),
        ArgumentException => nameof(ArgumentException),
        _ => nameof(InvalidOperationException)
    };

    private sealed record Options(string Command, string[] Arguments, string? Model, string? Language, string Format, string? Output, double Threshold);
    private sealed class UsageException(string message) : Exception(message);

    private static Options Parse(string[] args)
    {
        string command = args[0];
        if (command == "models" && args.Length > 1) command += " " + args[1];
        var positional = new List<string>();
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        bool literals = false;
        for (int i = command.StartsWith("models ", StringComparison.Ordinal) ? 2 : 1; i < args.Length; i++)
        {
            if (!literals && args[i] == "--") { literals = true; continue; }
            if (!literals && args[i].StartsWith("--", StringComparison.Ordinal))
            {
                string key = args[i];
                if (key is not ("--model" or "--language" or "--format" or "--output" or "--threshold")) throw new UsageException($"Unknown option '{key}'.");
                if (++i == args.Length || args[i].StartsWith("--", StringComparison.Ordinal)) throw new UsageException($"Missing value for '{key}'.");
                if (!values.TryAdd(key, args[i])) throw new UsageException($"Repeated option '{key}'.");
            }
            else positional.Add(args[i]);
        }
        int count = positional.Count;
        bool valid = command switch
        {
            "transcribe" or "diarize" or "detect-language" or "models pull" => count == 1,
            "align" or "enroll" or "verify" => count == 2,
            "identify" => count >= 3 && count % 2 == 1,
            "models list" => count == 0,
            _ => false
        };
        if (!valid) throw new UsageException("Unknown command or incorrect arguments.");
        bool document = command is "transcribe" or "align" or "diarize";
        bool speaker = command is "enroll" or "identify" or "verify";
        foreach (string key in values.Keys)
        {
            if ((key == "--format" && !document) || (key == "--threshold" && command != "identify") ||
                (key == "--language" && !(document || speaker)) ||
                (key == "--model" && !(document || speaker || command == "detect-language")))
                throw new UsageException($"'{key}' is not supported by '{command}'.");
        }
        string format = values.GetValueOrDefault("--format", "text");
        if (format is not ("text" or "json" or "srt" or "vtt")) throw new UsageException("Unknown output format.");
        double threshold = 0.3;
        if (values.TryGetValue("--threshold", out string? raw) && (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out threshold) || !double.IsFinite(threshold) || threshold < -1 || threshold > 1))
            throw new UsageException("Threshold must be a finite cosine in [-1,1].");
        return new(command, positional.ToArray(), values.GetValueOrDefault("--model"), values.GetValueOrDefault("--language"), format, values.GetValueOrDefault("--output"), threshold);
    }

    private static async ValueTask<string> ExecuteAsync(Options o, CancellationToken ct)
    {
        string[] a = o.Arguments;
        if (o.Command == "models list") return string.Join('\n', ModelStore.Default.Catalog.Entries.Select(m => $"{m.Id}\t{m.Tier}\t{m.License}\t{m.Revision}"));
        if (o.Command == "models pull") return (await ModelStore.Default.PullAsync(a[0], cancellationToken: ct)).Directory;
        if (o.Command == "detect-language") return Speech.DetectLanguage(a[0], ct, model: o.Model);
        Transcript document;
        if (o.Command == "transcribe") document = Speech.Transcribe(a[0], ct, o.Language, o.Model);
        else if (o.Command == "align") document = Speech.Align(a[0], await File.ReadAllTextAsync(a[1], ct), ct, o.Language, o.Model);
        else if (o.Command == "diarize")
        {
            using var source = AudioFileSource.Open(a[0], ct);
            long frames = 0;
            IEnumerable<AudioBuffer> Blocks()
            {
                var buffer = new float[16000 * source.Channels];
                int read;
                while ((read = source.Read(buffer, ct)) != 0)
                {
                    frames += read;
                    yield return new AudioBuffer(buffer.AsMemory(0, read * source.Channels), 16000, source.Channels);
                }
            }
            if (o.Model == "nvidia/Nemotron-3-Diarization")
            {
                if (o.Language is not null) throw new CapabilityNotSupportedException("Nemotron diarization has no language option.");
                using var diarizer = await NemotronDiarizer.LoadAsync(cancellationToken: ct);
                document = new Transcript { Text = "", Turns = diarizer.DiarizeBlocks(Blocks(), ct) };
            }
            else
            {
                using var embedder = await LoadEmbedderAsync(o, ct);
                using var vad = new SileroVad(cancellationToken: ct);
                var diarizer = new ClusteringDiarizer(embedder, vad);
                document = new Transcript { Text = "", Turns = diarizer.DiarizeBlocks(Blocks(), ct), Models = diarizer.Models };
            }
            document = document with
            {
                Duration = TimeSpan.FromSeconds((double)frames / 16000),
                Segments = document.Turns.Select(t => new Segment { Text = t.Speaker.Id, Start = t.Start, End = t.End }).ToImmutableArray()
            };
            if (o.Format == "text") return string.Join('\n', document.Turns.Select(t => string.Create(CultureInfo.InvariantCulture, $"{t.Start.TotalSeconds:F3}\t{t.End.TotalSeconds:F3}\t{t.Speaker.Id}")));
        }
        else
        {
            using var embedder = await LoadEmbedderAsync(o, ct);
            using var registry = new InMemorySpeakerRegistry(embedder, o.Threshold);
            if (o.Command == "verify") return registry.Verify(ReadAudio(a[0], ct), ReadAudio(a[1], ct), ct).ToString("R", CultureInfo.InvariantCulture);
            if (o.Command == "enroll") { registry.Enroll(a[0], ReadAudio(a[1], ct), ct); return $"Enrolled '{a[0]}' in memory for this invocation only; enrollment discarded on exit."; }
            for (int i = 1; i < a.Length; i += 2) registry.Enroll(a[i], ReadAudio(a[i + 1], ct), ct);
            var match = registry.Identify(ReadAudio(a[0], ct), ct);
            return $"{match.SpeakerId ?? "unknown"}\t{match.Score.ToString("R", CultureInfo.InvariantCulture)}";
        }
        return o.Format switch { "json" => document.ToJson(), "srt" => document.ToSrt(ct), "vtt" => document.ToVtt(ct), _ => document.Text };
    }

    private static async ValueTask<CampPlusEmbedder> LoadEmbedderAsync(Options o, CancellationToken ct)
    {
        if (o.Language is not (null or "en" or "zh")) throw new CapabilityNotSupportedException("Speaker commands support --language en or zh.");
        string id = o.Model ?? (o.Language == "zh" ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k");
        if (id is not ("iic/speech_campplus_sv_en_voxceleb_16k" or "iic/speech_campplus_sv_zh-cn_16k-common")) throw new CapabilityNotSupportedException("Select a CAM++ embedder, or use --model nvidia/Nemotron-3-Diarization for diarize.");
        var resolved = await ModelStore.Default.PullAsync(id, cancellationToken: ct);
        bool chinese = id.EndsWith("common", StringComparison.Ordinal);
        return new CampPlusEmbedder(Path.Combine(resolved.Directory, chinese ? "campplus-zh-fp32.onemodel" : "campplus-en-fp32.onemodel"), resolved.Info, chinese ? 192 : 512, cancellationToken: ct);
    }

    private static AudioBuffer ReadAudio(string path, CancellationToken ct)
    {
        using var source = AudioFileSource.Open(path, ct);
        var samples = new List<float>();
        var block = new float[16000 * source.Channels];
        int read;
        while ((read = source.Read(block, ct)) != 0)
        {
            if (samples.Count / source.Channels + read > 60 * 16000) throw new CapabilityNotSupportedException("Enrollment and verification audio is limited to 60 seconds per file.");
            samples.AddRange(block.AsSpan(0, read * source.Channels).ToArray());
        }
        float[] mono = new float[samples.Count / source.Channels];
        for (int i = 0; i < mono.Length; i++) for (int c = 0; c < source.Channels; c++) mono[i] += samples[i * source.Channels + c] / source.Channels;
        return new AudioBuffer(mono, 16000);
    }
}
