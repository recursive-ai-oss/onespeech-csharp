using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Whisper;

// Atomic per-file outputs allow bounded runs to resume without redoing inference.
internal static class FileComparison
{
    internal static int PrepareClips(string input, string output)
    {
        var audio = OneAudio.Audio.Load(input, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        Directory.CreateDirectory(output);
        foreach (int seconds in new[] { 60, 600 })
        {
            using var stream = File.Create(Path.Combine(output, seconds + ".wav"));
            using var encoder = new OneAudio.WavEncoder(stream, new OneAudio.AudioFormat(16000, 1));
            encoder.Write(audio.Samples.Span[..(seconds * 16000)]); encoder.Finish();
        }
        return 0;
    }
    internal static int PrepareEarnings(string root, string output)
    {
        // The second recording is chosen by numeric ID order, independently of scores.
        string second = Directory.GetFiles(Path.Combine(root, "media"), "*.mp3").Select(Path.GetFileNameWithoutExtension).Where(id => long.TryParse(id, out _)).Order(StringComparer.Ordinal).First()!;
        using var writer = new StreamWriter(output);
        foreach (string id in new[] { "4474955", second })
        {
            string text = string.Join(' ', File.ReadLines(Path.Combine(root, "transcripts", "nlp_references", id + ".nlp")).Skip(1).Select(l => l.Split('|')[0]));
            using var memory = new MemoryStream();
            using (var json = new Utf8JsonWriter(memory)) { json.WriteStartObject(); json.WriteString("id", id); json.WriteString("audio", Path.Combine(root, "media", id + ".mp3")); json.WriteString("text", text); json.WriteEndObject(); }
            writer.WriteLine(Encoding.UTF8.GetString(memory.ToArray()));
        }
        return 0;
    }
    internal static int Prepare(string root, string output)
    {
        using var writer = new StreamWriter(output);
        foreach (string path in Directory.EnumerateFiles(root, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileName, StringComparer.Ordinal).Take(200))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            string text = File.ReadLines(Directory.GetFiles(Path.GetDirectoryName(path)!, "*.trans.txt").Single()).Single(l => l.StartsWith(id + " ", StringComparison.Ordinal))[(id.Length + 1)..];
            using var memory = new MemoryStream();
            using (var json = new Utf8JsonWriter(memory)) { json.WriteStartObject(); json.WriteString("id", id); json.WriteString("audio", path); json.WriteString("text", text); json.WriteEndObject(); }
            writer.WriteLine(Encoding.UTF8.GetString(memory.ToArray()));
        }
        return 0;
    }
    internal static int Run(string[] args)
    {
        string mode = args[0], manifest = args[1], output = args[2], model = args[3];
        int begin = int.Parse(args[4], CultureInfo.InvariantCulture), count = int.Parse(args[5], CultureInfo.InvariantCulture);
        Directory.CreateDirectory(output);
        using var recognizer = mode == "comparison-greedy" ? WhisperRecognizer.Load(model, new WhisperOptions { Threads = 8, Temperatures = [0], WordTimestamps = false }) : null;
        foreach (string line in File.ReadLines(manifest).Skip(begin).Take(count))
        {
            using var doc = JsonDocument.Parse(line); var row = doc.RootElement;
            string id = row.GetProperty("id").GetString()!, destination = Path.Combine(output, id + ".json");
            if (File.Exists(destination)) continue;
            string audioPath = row.GetProperty("audio").GetString()!;
            var clock = Stopwatch.StartNew();
            Transcript transcript;
            if (mode == "comparison-speed") transcript = SpeedPipeline(audioPath, model, Environment.ProcessorCount);
            else if (recognizer is null) transcript = Speech.Transcribe(audioPath, language: "en", model: model);
            else
            {
                using var input = AudioFileSource.Open(audioPath, CancellationToken.None);
                List<float> samples = []; float[] block = new float[4096]; int n;
                while ((n = input.Read(block, CancellationToken.None)) != 0) samples.AddRange(block.AsSpan(0, n).ToArray());
                transcript = recognizer.Transcribe(new AudioBuffer(samples.ToArray(), 16000), "en", CancellationToken.None);
            }
            File.WriteAllText(destination + ".part", transcript.ToJson()); File.Move(destination + ".part", destination);
            using var process = Process.GetCurrentProcess();
            File.WriteAllText(destination + ".measurement.json", FormattableString.Invariant($"{{\"seconds\":{clock.Elapsed.TotalSeconds:R},\"peakRss\":{process.PeakWorkingSet64},\"processorCount\":{Environment.ProcessorCount}}}"));
            Console.WriteLine(id);
        }
        using var hypotheses = new StreamWriter(Path.Combine(output, "hypotheses.jsonl"), false, new UTF8Encoding(false));
        using var truth = new StreamWriter(Path.Combine(output, "truth.jsonl"), false, new UTF8Encoding(false));
        foreach (string line in File.ReadLines(manifest))
        {
            using var doc = JsonDocument.Parse(line); var row = doc.RootElement; string id = row.GetProperty("id").GetString()!;
            string path = Path.Combine(output, id + ".json"); if (!File.Exists(path)) continue;
            var transcript = Transcript.FromJson(File.ReadAllText(path));
            Write(hypotheses, id, transcript.Text, transcript.Words.Select(w => (w.Text, w.Start?.TotalSeconds, w.End?.TotalSeconds)));
            Write(truth, id, row.GetProperty("text").GetString()!, []);
        }
        return 0;
    }
    private static Transcript SpeedPipeline(string path, string model, int threads)
    {
        var store = OneSpeech.Catalog.ModelStore.Default;
        var resolved = store.PullAsync(model).AsTask().GetAwaiter().GetResult();
        using var qwenAligner = model.StartsWith("Qwen/", StringComparison.Ordinal) ? OneSpeech.Qwen3.Qwen3ForcedAligner.Load(store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B").AsTask().GetAwaiter().GetResult().Directory, threads) : null;
        using var whisper = qwenAligner is null ? WhisperRecognizer.Load(resolved.Directory, new WhisperOptions { Threads = threads }) : null;
        using var qwen = qwenAligner is not null ? OneSpeech.Qwen3.Qwen3Recognizer.Load(resolved.Directory, new OneSpeech.Qwen3.Qwen3Options { Threads = threads }) : null;
        SpeechRecognizer recognizer = (SpeechRecognizer?)whisper ?? new TimedRecognizer(qwen!, qwenAligner!);
        var ctc = store.PullAsync("facebook/omniASR-CTC-300M").AsTask().GetAwaiter().GetResult();
        using var timing = new OneSpeech.Alignment.OmniAsrAligner(Path.Combine(ctc.Directory, "ctc-fp32.onemodel"), Path.Combine(ctc.Directory, "omniASR_tokenizer.model"), ctc.Info, threads);
        using var vad = new OneSpeech.Vad.SileroVad();
        var transcript = new OneSpeech.Pipeline.VadTranscriptionPipeline(recognizer, vad) { TimingAligners = [timing] }.Transcribe(path, "en", CancellationToken.None);
        var speaker = store.PullAsync("iic/speech_campplus_sv_en_voxceleb_16k").AsTask().GetAwaiter().GetResult();
        using var embedder = new OneSpeech.Embedders.CampPlusEmbedder(Path.Combine(speaker.Directory, store.Catalog.Get(speaker.Info.Id).ConvertedFiles.Single().Name), speaker.Info, threads: threads);
        using var activity = new OneSpeech.Vad.SileroVad(new OneSpeech.Vad.SileroVadOptions { Threshold = 0.2f, NegativeThreshold = 0.2f - 0.15f, MinimumSpeech = TimeSpan.FromMilliseconds(250), MinimumSilence = TimeSpan.FromMilliseconds(250), SpeechPadding = TimeSpan.FromMilliseconds(150) });
        var diarizer = new OneSpeech.Diarization.ClusteringDiarizer(embedder, activity, resegmenter: new OneSpeech.Diarization.CosineResegmenter(0.4, 2));
        using var source = AudioFileSource.Open(path, CancellationToken.None);
        var turns = diarizer.DiarizeBlocks(Blocks(source), CancellationToken.None);
        return OneSpeech.Diarization.TranscriptDiarization.Merge(transcript, turns.Select(t => new OneSpeech.Diarization.ActiveTurn { Turn = t }), CancellationToken.None);
    }
    private sealed class TimedRecognizer(SpeechRecognizer recognizer, Aligner aligner) : SpeechRecognizer
    {
        public override ModelInfo Model => recognizer.Model;
        public override Capabilities Capabilities => recognizer.Capabilities;
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => recognizer.DetectLanguage(audio, cancellationToken);
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            var transcript = recognizer.Transcribe(audio, language, cancellationToken);
            var aligned = aligner.Align(audio, transcript.Text, language, cancellationToken);
            return aligned with { Models = [Model, .. aligned.Models] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => Transcribe(audio, language, cancellationToken).Segments;
    }
    private static IEnumerable<AudioBuffer> Blocks(AudioFileSource source)
    {
        float[] block = new float[4096 * source.Channels]; int count;
        while ((count = source.Read(block, CancellationToken.None)) != 0) yield return new AudioBuffer(block.AsMemory(0, count * source.Channels), 16000, source.Channels);
    }
    private static void Write(StreamWriter writer, string id, string text, IEnumerable<(string Text, double? Start, double? End)> words)
    {
        using var memory = new MemoryStream();
        using (var json = new Utf8JsonWriter(memory))
        {
            json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", text); json.WriteStartArray("words");
            foreach (var word in words)
            {
                json.WriteStartObject(); json.WriteString("text", word.Text);
                if (word.Start is { } start && word.End is { } end) { json.WriteNumber("start", start); json.WriteNumber("end", end); }
                else { json.WriteNull("start"); json.WriteNull("end"); }
                json.WriteEndObject();
            }
            json.WriteEndArray(); json.WriteEndObject();
        }
        writer.WriteLine(Encoding.UTF8.GetString(memory.ToArray()));
    }
}
