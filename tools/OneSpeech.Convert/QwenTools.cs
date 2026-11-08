using System.Security.Cryptography;
using OneSpeech.Formats;

internal static class QwenTools
{
    internal static void CachePaths(string root)
    {
        foreach (string id in new[] { "Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B" })
        {
            var entry = OneSpeech.Catalog.ModelCatalog.Default.Get(id);
            string key = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(entry.Id + "\n" + entry.Revision + "\n" + string.Join('\n', entry.Files.Select(file => file.Name + ":" + file.Sha256)))));
            Console.WriteLine(id + "\t" + Path.Combine(root, key));
        }
    }
    internal static void EndToEnd(string cache, string audio, string output)
    {
        OneSpeech.Catalog.ModelStore.Default = new OneSpeech.Catalog.ModelStore(cache);
        var transcript = OneSpeech.Speech.Transcribe(audio, model: "Qwen/Qwen3-ASR-0.6B");
        if (transcript.Text.Length == 0 || transcript.Language != "en" || transcript.Words.IsEmpty || transcript.Turns.IsEmpty ||
            !transcript.Models.Any(model => model.Id == "Qwen/Qwen3-ForcedAligner-0.6B") ||
            transcript.Words.Any(word => word.Start == null || word.End == null || word.Confidence == null || word.TimingMethod != OneSpeech.TimingMethod.ForcedAligner))
            throw new InvalidDataException("Qwen Speech.Transcribe document is incomplete.");
        File.WriteAllText(output, transcript.ToJson());
        Console.WriteLine("Speech.Transcribe Qwen end-to-end: language=" + transcript.Language + "; words=" + transcript.Words.Length + "; turns=" + transcript.Turns.Length + "; text=" + transcript.Text);
    }
    internal static void CanonicalAudio(string audio, string output)
    {
        var clip = OneAudio.Audio.Load(audio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        using var writer = new BinaryWriter(File.Create(output));
        foreach (float sample in clip.Samples.Span) writer.Write(sample);
        Console.WriteLine("Canonical 16kHz mono frames: " + clip.Samples.Length);
    }
    internal static void LongForm(string model, string audio, string output, bool bf16)
    {
        using var recognizer = OneSpeech.Qwen3.Qwen3Recognizer.Load(model, new OneSpeech.Qwen3.Qwen3Options { Threads = 4, Precision = bf16 ? OneSpeech.Qwen3.Qwen3Precision.BFloat16 : OneSpeech.Qwen3.Qwen3Precision.Binary32 });
        using var vad = new OneSpeech.Vad.SileroVad();
        // The file pipeline uses these same bounded VAD chunks. Checkpoint each completed
        // recognition call so a bounded evaluation job can resume without changing chunking.
        string checkpoints = output + ".chunks";
        Directory.CreateDirectory(checkpoints);
        using (var input = File.OpenRead(audio))
        {
            string fingerprint = Convert.ToHexStringLower(SHA256.HashData(input)) + "\n" + recognizer.Model.Id + "\n" + bf16;
            string manifest = Path.Combine(checkpoints, "input.txt");
            if (File.Exists(manifest) && File.ReadAllText(manifest) != fingerprint) throw new InvalidDataException("Long-form checkpoint input differs.");
            File.WriteAllText(manifest, fingerprint);
        }
        using var source = OneSpeech.Audio.AudioFileSource.Open(audio, CancellationToken.None);
        var chunker = new OneSpeech.Audio.VadChunker(vad);
        var texts = new List<string>();
        int index = 0;
        foreach (var chunk in chunker.Read(OneSpeech.Audio.VadChunker.Blocks(source, CancellationToken.None), null, CancellationToken.None))
        {
            string path = Path.Combine(checkpoints, index.ToString("D5", System.Globalization.CultureInfo.InvariantCulture) + ".json");
            if (!File.Exists(path))
            {
                var recognized = recognizer.Transcribe(chunk.Audio, "en", CancellationToken.None);
                File.WriteAllText(path + ".partial", recognized.ToJson());
                File.Move(path + ".partial", path);
            }
            string text = OneSpeech.Transcript.FromJson(File.ReadAllText(path)).Text;
            if (text.Length != 0) texts.Add(text);
            Console.WriteLine("Chunk " + index + " at audio seconds " + chunk.Start.TotalSeconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
            index++;
        }
        var result = new OneSpeech.Transcript { Text = string.Join(' ', texts), Models = [recognizer.Model], Language = "en" };
        File.WriteAllText(output, result.ToJson());
        Console.WriteLine("Long-form Qwen chunks=" + index);
    }
    private static string LoadAverage() => File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : "unavailable";
    private static void Metadata(string output, string model, string precision, int begin, string before, string during, string status)
    {
        using var stream = File.Create(Path.Combine(output, "metadata-" + begin + ".json"));
        using var writer = new System.Text.Json.Utf8JsonWriter(stream);
        writer.WriteStartObject(); writer.WriteString("status", status); writer.WriteString("modelDirectory", model); writer.WriteString("precision", precision);
        writer.WriteNumber("threads", 4); writer.WriteNumber("runtimeProcessorCount", Environment.ProcessorCount);
        writer.WriteString("runtime", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
        writer.WriteString("os", System.Runtime.InteropServices.RuntimeInformation.OSDescription);
        writer.WriteString("oneModelAssembly", typeof(RecursiveAI.OneModel.Graph.Session).Assembly.GetName().Version?.ToString());
        writer.WriteString("oneAudioAssembly", typeof(OneAudio.Audio).Assembly.GetName().Version?.ToString());
        writer.WriteString("loadBefore", before); writer.WriteString("loadDuring", during); writer.WriteString("loadAfter", LoadAverage());
        writer.WriteEndObject();
    }
    internal static void Rebuild(string directory, bool bf16)
    {
        string staging = Path.Combine(directory, "build");
        OneSpeech.Catalog.Conversion.QwenConverter.Convert(directory, staging, CancellationToken.None,
            bf16 ? OneSpeech.Qwen3.Qwen3Precision.BFloat16 : OneSpeech.Qwen3.Qwen3Precision.Binary32);
        foreach (string file in Directory.EnumerateFiles(staging, "*.onemodel"))
            File.Move(file, Path.Combine(directory, Path.GetFileName(file)), overwrite: true);
        Directory.Delete(staging);
    }
    internal static void Reproduce(string directory, bool bf16)
    {
        string staging = Path.Combine(directory, "reproduce");
        OneSpeech.Catalog.Conversion.QwenConverter.Convert(directory, staging, CancellationToken.None,
            bf16 ? OneSpeech.Qwen3.Qwen3Precision.BFloat16 : OneSpeech.Qwen3.Qwen3Precision.Binary32);
        static (string Manifest, string DataHash) Contents(string path)
        {
            using var input = File.OpenRead(path);
            byte[] header = new byte[64]; input.ReadExactly(header);
            long length = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(header.AsSpan(24));
            if (length is < 1 or > 8388608) throw new InvalidDataException("Graph manifest exceeds proof bound.");
            byte[] manifest = new byte[(int)length]; input.ReadExactly(manifest);
            input.Position = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(header.AsSpan(32));
            return (System.Text.Encoding.UTF8.GetString(manifest), Convert.ToHexStringLower(SHA256.HashData(input)));
        }
        foreach (string file in Directory.EnumerateFiles(staging, "*.onemodel").Order(StringComparer.Ordinal))
        {
            string target = Path.Combine(directory, Path.GetFileName(file));
            if (File.Exists(target))
            {
                using var old = File.OpenRead(target); using var current = File.OpenRead(file);
                string expected = Convert.ToHexStringLower(SHA256.HashData(current));
                if (Convert.ToHexStringLower(SHA256.HashData(old)) != expected)
                {
                    var before = Contents(target); var after = Contents(file);
                    if (before.Manifest.Replace("\"name\":\"Whisper\"", "\"name\":\"Qwen3\"", StringComparison.Ordinal) != after.Manifest || before.DataHash != after.DataHash)
                        throw new InvalidDataException("Reproduced graph computation differs: " + Path.GetFileName(file));
                    Console.WriteLine("Metadata-only graph name update: " + Path.GetFileName(file));
                }
                else Console.WriteLine("Reproduced " + Path.GetFileName(file) + " " + expected);
            }
            File.Move(file, target, overwrite: true);
        }
        Directory.Delete(staging);
    }
    internal static void AlignBatch(string model, string root, string split, string output, int begin, int count)
    {
        Directory.CreateDirectory(output);
        using var aligner = OneSpeech.Qwen3.Qwen3ForcedAligner.Load(model, 4);
        var references = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string file in Directory.EnumerateFiles(Path.Combine(root, split), "*.trans.txt", SearchOption.AllDirectories))
            foreach (string line in File.ReadLines(file)) { int space = line.IndexOf(' '); references.Add(line[..space], line[(space + 1)..]); }
        var paths = Directory.EnumerateFiles(Path.Combine(root, split), "*.flac", SearchOption.AllDirectories)
            .OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(200).Skip(begin).Take(count);
        foreach (string path in paths)
        {
            string id = Path.GetFileNameWithoutExtension(path), target = Path.Combine(output, id + ".json");
            if (File.Exists(target)) continue;
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var result = aligner.Align(new OneSpeech.AudioBuffer(clip.Samples, 16000), references[id], "en", CancellationToken.None);
            using (var stream = File.Create(target + ".partial"))
            using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
            {
                writer.WriteStartObject(); writer.WriteString("id", id); writer.WriteString("text", result.Text); writer.WriteStartArray("words");
                foreach (var word in result.Words)
                {
                    writer.WriteStartObject(); writer.WriteString("text", OneSpeech.Qwen3.Qwen3ForcedAligner.SpokenForm(word.Text)); writer.WriteString("written", word.Text);
                    if (word.Start is { } start) writer.WriteNumber("start", start.TotalSeconds); else writer.WriteNull("start");
                    if (word.End is { } end) writer.WriteNumber("end", end.TotalSeconds); else writer.WriteNull("end");
                    writer.WriteEndObject();
                }
                writer.WriteEndArray(); writer.WriteEndObject();
            }
            File.Move(target + ".partial", target); Console.WriteLine(id);
        }
    }
    internal static void Catalog(string directory, string output)
    {
        string[] provenance = File.ReadAllLines(Path.Combine(directory, "upstream.txt"));
        using var file = File.Create(output);
        using var writer = new System.Text.Json.Utf8JsonWriter(file, new System.Text.Json.JsonWriterOptions { Indented = true });
        writer.WriteStartObject(); writer.WriteString("id", provenance[0]);
        bool aligner = provenance[0].Contains("ForcedAligner", StringComparison.Ordinal);
        writer.WriteString("family", aligner ? "qwen3-aligner" : "qwen3-asr");
        writer.WriteString("source", "https://huggingface.co/" + provenance[0]); writer.WriteString("revision", provenance[1]);
        void WriteFiles(string property, IEnumerable<string> paths, bool upstream)
        {
            writer.WriteStartArray(property);
            foreach (string path in paths.Order(StringComparer.Ordinal))
            {
                string name = Path.GetFileName(path);
                using var stream = File.OpenRead(path);
                writer.WriteStartObject(); writer.WriteString("name", name); writer.WriteNumber("size", stream.Length);
                writer.WriteString("sha256", Convert.ToHexStringLower(SHA256.HashData(stream)));
                if (upstream) writer.WriteString("upstreamUrl", "https://huggingface.co/" + provenance[0] + "/resolve/" + provenance[1] + "/" + name);
                else writer.WriteString("upstreamUrl", "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/models/" + provenance[0].Replace('/', '-') + "-" + name);
                writer.WriteNull("mirrorUrl"); writer.WriteEndObject();
            }
            writer.WriteEndArray();
        }
        WriteFiles("files", Directory.EnumerateFiles(directory).Where(path => Path.GetExtension(path) is ".json" or ".safetensors" || Path.GetFileName(path) == "merges.txt"), true);
        WriteFiles("convertedFiles", Directory.EnumerateFiles(directory, "*.onemodel"), false);
        writer.WriteBoolean("isDefault", false); writer.WriteString("tier", "Tier1"); writer.WriteString("license", "Apache-2.0");
        writer.WriteString("attribution", aligner ? "Qwen Team, Qwen3-ForcedAligner, Apache License 2.0." : "Qwen Team, Qwen3-ASR, Apache License 2.0.");
        writer.WriteBoolean("gated", false); writer.WriteStartObject("capabilities"); writer.WriteStartArray("languages");
        foreach (string language in aligner ? new[] { "zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es" } : OneSpeech.Qwen3.Qwen3Languages.Tags.ToArray()) writer.WriteStringValue(language);
        writer.WriteEndArray(); writer.WriteBoolean("biasing", !aligner); writer.WriteBoolean("nativeTimestamps", aligner); writer.WriteEndObject(); writer.WriteEndObject();
    }
    internal static void DecodeIds(string directory, string path)
    {
        var tokenizer = new OneSpeech.Qwen3.Qwen3Tokenizer(directory);
        using var reader = new BinaryReader(File.OpenRead(path));
        List<int> tokens = [];
        while (reader.BaseStream.Position < reader.BaseStream.Length) tokens.Add(checked((int)reader.ReadInt64()));
        Console.WriteLine(tokenizer.Decode(tokens.ToArray(), true));
        Console.WriteLine(string.Join(',', tokens));
    }
    internal static void Batch(string model, string root, string split, string output, int begin, int count, OneSpeech.Qwen3.Qwen3Precision precision = OneSpeech.Qwen3.Qwen3Precision.Binary32)
    {
        Directory.CreateDirectory(output);
        string loadBefore = LoadAverage(), loadDuring = "unavailable";
        Metadata(output, model, precision.ToString(), begin, loadBefore, loadDuring, "running");
        using var recognizer = OneSpeech.Qwen3.Qwen3Recognizer.Load(model, new OneSpeech.Qwen3.Qwen3Options { Threads = 4, Precision = precision });
        var paths = Directory.EnumerateFiles(Path.Combine(root, split), "*.flac", SearchOption.AllDirectories)
            .OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(200).Skip(begin).Take(count);
        foreach (string path in paths)
        {
            string id = Path.GetFileNameWithoutExtension(path), target = Path.Combine(output, id + ".json");
            if (File.Exists(target)) continue;
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var result = recognizer.Transcribe(new OneSpeech.AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
            if (loadDuring == "unavailable") { loadDuring = LoadAverage(); Metadata(output, model, precision.ToString(), begin, loadBefore, loadDuring, "running"); }
            using (var stream = File.Create(target + ".partial"))
            using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
            {
                writer.WriteStartObject(); writer.WriteString("id", id); writer.WriteString("text", result.Text); writer.WriteString("language", result.Language); writer.WriteEndObject();
            }
            File.Move(target + ".partial", target);
            Console.WriteLine(id);
        }
        Metadata(output, model, precision.ToString(), begin, loadBefore, loadDuring, "complete");
    }
    internal static void ProbeDecoder(string directory, string probe, bool bf16 = false)
    {
        static float[] Read(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            var values = new float[reader.BaseStream.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadSingle();
            return values;
        }
        using var session = new OneSpeech.Qwen3.QwenModelSession(directory, 4, CancellationToken.None, bf16 ? OneSpeech.Qwen3.Qwen3Precision.BFloat16 : OneSpeech.Qwen3.Qwen3Precision.Binary32);
        var tokenizer = new OneSpeech.Qwen3.Qwen3Tokenizer(directory);
        using var reader = new BinaryReader(File.OpenRead(Path.Combine(probe, session.Configuration.TimestampClasses > 0 ? "input_ids.i64" : "forward-input_ids.i64")));
        int[] prompt = new int[reader.BaseStream.Length / 8];
        for (int i = 0; i < prompt.Length; i++) prompt[i] = checked((int)reader.ReadInt64());
        Console.WriteLine("Prompt: " + tokenizer.Decode(prompt, true));
        using var encoded = session.Encode(new OneSpeech.AudioBuffer(Read(Path.Combine(probe, "audio.f32")), 16000), CancellationToken.None);
        RecursiveAI.OneModel.Graph.SessionOutputs[]? past = null;
        try
        {
            float[] logits = session.Decode(prompt, encoded["encoder"], ref past, 0, CancellationToken.None);
            string logitPath = Path.Combine(probe, "lm-head-output.f32");
            if (File.Exists(logitPath))
            {
                float[] expectedLogits = Read(logitPath);
                float maxLogitError = 0, referenceScale = 1;
                for (int i = 0; i < logits.Length; i++) { maxLogitError = Math.Max(maxLogitError, Math.Abs(logits[i] - expectedLogits[i])); referenceScale = Math.Max(referenceScale, Math.Abs(expectedLogits[i])); }
                Console.WriteLine("logits max_abs=" + maxLogitError.ToString("G9", System.Globalization.CultureInfo.InvariantCulture) + "; scaled=" + (maxLogitError / referenceScale).ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
            }
            float[] reference = Read(Path.Combine(probe, session.Configuration.TimestampClasses > 0 ? "classifier-input.f32" : "decoder-final-norm-output.f32"));
            var hidden = past![^1]["decoder"].Data<float>();
            float error = 0;
            for (int i = 0; i < reference.Length; i++) error = Math.Max(error, Math.Abs(hidden[i] - reference[i]));
            Console.WriteLine("decoder norm max_abs=" + error.ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
            if (session.Configuration.TimestampClasses > 0)
            {
                for (int position = 0; position < prompt.Length; position++)
                {
                    if (prompt[position] != 151705) continue;
                    int classes = session.Configuration.TimestampClasses, best = 0;
                    for (int i = 1; i < classes; i++) if (logits[position * classes + i] > logits[position * classes + best]) best = i;
                    Console.WriteLine("timestamp=" + (best * .08).ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
                }
                return;
            }
            List<int> generated = [];
            for (int step = 0; step < 128; step++)
            {
                int best = 0;
                for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
                if (best == 151645) break;
                generated.Add(best);
                logits = session.Decode([best], null, ref past, prompt.Length + step, CancellationToken.None);
            }
            Console.WriteLine("Generated: " + tokenizer.Decode(generated.ToArray(), true));
            string generatedPath = Path.Combine(probe, "generated_ids.i64");
            if (File.Exists(generatedPath))
            {
                using var referenceIds = new BinaryReader(File.OpenRead(generatedPath));
                var expected = new List<int>();
                while (referenceIds.BaseStream.Position < referenceIds.BaseStream.Length) expected.Add(checked((int)referenceIds.ReadInt64()));
                int[] suffix = expected.Skip(prompt.Length).TakeWhile(id => id != 151645).ToArray();
                Console.WriteLine("Greedy token identity: " + generated.SequenceEqual(suffix) + "; ours=" + generated.Count + "; reference=" + suffix.Length);
            }
        }
        finally { if (past != null) foreach (var output in past) output.Dispose(); }
    }
    internal static void ProbeEncoder(string directory, string probe)
    {
        static float[] Read(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            var values = new float[reader.BaseStream.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadSingle();
            return values;
        }
        static void Compare(string name, ReadOnlySpan<float> actual, float[] expected)
        {
            if (actual.Length != expected.Length) throw new InvalidDataException("Qwen parity shape mismatch: " + name);
            float error = 0; double squared = 0;
            for (int i = 0; i < actual.Length; i++) { float delta = Math.Abs(actual[i] - expected[i]); error = Math.Max(error, delta); squared += (double)delta * delta; }
            Console.WriteLine($"{name}: count={actual.Length} max_abs={error:G9} rms={Math.Sqrt(squared / actual.Length):G9}");
        }
        var config = OneSpeech.Qwen3.QwenConfiguration.Load(directory);
        float[] audio = Read(Path.Combine(probe, "audio.f32"));
        float[] mel = new OneSpeech.Audio.WhisperLogMelSpectrogram(128).Extract(new OneSpeech.AudioBuffer(audio, 16000), false);
        int frames = mel.Length / 128, chunks = (frames + 99) / 100;
        float[] convolutionInput = new float[chunks * 128 * 100];
        for (int chunk = 0; chunk < chunks; chunk++)
            for (int bin = 0; bin < 128; bin++)
                for (int frame = 0; frame < Math.Min(100, frames - chunk * 100); frame++)
                    convolutionInput[(chunk * 128 + bin) * 100 + frame] = mel[bin * frames + chunk * 100 + frame];
        var options = new RecursiveAI.OneModel.Graph.ModelOptions { IntraOpThreads = 4 };
        using var front = RecursiveAI.OneModel.Graph.Session.Load(Path.Combine(directory, "encoder-front.onemodel"), options);
        using var input = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(convolutionInput, RecursiveAI.OneModel.Tensors.Shape.Create(chunks, 1, 128, 100));
        using var convolved = front.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor>(StringComparer.Ordinal) { ["mel_chunks"] = input });
        var conv = convolved["conv"].Data<float>();
        Compare("conv", conv, Read(Path.Combine(probe, "conv_out-output.f32")));
        int sequence = (chunks - 1) * 13 + (frames - (chunks - 1) * 100 + 7) / 8;
        float[] hidden = new float[sequence * config.AudioWidth], positions = new float[sequence * config.AudioWidth], positioned = new float[sequence * config.AudioWidth];
        int half = config.AudioWidth / 2;
        for (int position = 0; position < sequence; position++)
            for (int dim = 0; dim < config.AudioWidth; dim++)
            {
                int step = position % 13;
                float frequency = MathF.Exp(-MathF.Log(10000) * (dim % half) / (half - 1));
                float angle = step * frequency;
                hidden[position * config.AudioWidth + dim] = conv[position * config.AudioWidth + dim];
                positions[position * config.AudioWidth + dim] = dim < half ? MathF.Sin(angle) : MathF.Cos(angle);
                positioned[position * config.AudioWidth + dim] = hidden[position * config.AudioWidth + dim] + positions[position * config.AudioWidth + dim];
            }
        Compare("encoder-input", positioned, Read(Path.Combine(probe, "layers.0.self_attn_layer_norm-input-0.f32")));
        float[] mask = new float[sequence * sequence];
        using var hiddenTensor = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(hidden, RecursiveAI.OneModel.Tensors.Shape.Create(1, sequence, config.AudioWidth));
        using var positionTensor = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(positions, RecursiveAI.OneModel.Tensors.Shape.Create(1, sequence, config.AudioWidth));
        using var maskTensor = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(mask, RecursiveAI.OneModel.Tensors.Shape.Create(1, 1, sequence, sequence));
        using var encoder = RecursiveAI.OneModel.Graph.Session.Load(Path.Combine(directory, "encoder.onemodel"), options);
        using var encoded = encoder.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor>(StringComparer.Ordinal) { ["hidden"] = hiddenTensor, ["positions"] = positionTensor, ["mask"] = maskTensor });
        Compare("normalized", encoded["normalized"].Data<float>(), Read(Path.Combine(probe, "ln_post-output.f32")));
        Compare("encoder", encoded["encoder"].Data<float>(), Read(Path.Combine(probe, "encoder-output.f32")));
    }
    internal static void ConvertEncoder(string directory) => OneSpeech.Catalog.Conversion.QwenConverter.Convert(directory, directory, CancellationToken.None);
    internal static void Positions(string directory)
    {
        static float[] Read(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            var values = new float[reader.BaseStream.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = reader.ReadSingle();
            return values;
        }
        float[] conv = Read(Path.Combine(directory, "conv_out-output.f32"));
        float[] hidden = Read(Path.Combine(directory, "layers.0.self_attn_layer_norm-input-0.f32"));
        foreach (int frame in new[] { 0, 1, 12, 13, 26, 39, 42 })
        {
            Console.Write("position " + frame + ": ");
            foreach (int dim in new[] { 0, 1, 100, 447, 448, 449, 895 })
                Console.Write((hidden[frame * 896 + dim] - conv[frame * 896 + dim]).ToString("G9", System.Globalization.CultureInfo.InvariantCulture) + " ");
            Console.WriteLine();
        }
    }
    internal static async Task FetchAsync(string name, string directory, CancellationToken cancellationToken)
    {
        var pin = name switch
        {
            "0.6b" => ("Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", new[] { ("model.safetensors", "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea") }),
            "1.7b" => ("Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", new[] { ("model-00001-of-00002.safetensors", "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"), ("model-00002-of-00002.safetensors", "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc") }),
            "aligner" => ("Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7", new[] { ("model.safetensors", "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7") }),
            _ => throw new ArgumentException("Expected 0.6b, 1.7b or aligner.", nameof(name))
        };
        Directory.CreateDirectory(directory);
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
        async Task Fetch(string file, string? hash)
        {
            string target = Path.Combine(directory, file);
            if (File.Exists(target))
            {
                if (hash is null) return;
                await using var existing = File.OpenRead(target);
                if (Convert.ToHexStringLower(await SHA256.HashDataAsync(existing, cancellationToken).ConfigureAwait(false)) == hash) return;
                throw new InvalidDataException("Cached Qwen weight SHA-256 mismatch.");
            }
            string partial = target + ".partial";
            try
            {
                using var response = await client.GetAsync("https://huggingface.co/Qwen/" + pin.Item1 + "/resolve/" + pin.Item2 + "/" + file, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();
                await using (var destination = File.Create(partial))
                await using (var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                    await source.CopyToAsync(destination, cancellationToken).ConfigureAwait(false);
                if (hash is not null)
                {
                    await using var verify = File.OpenRead(partial);
                    if (Convert.ToHexStringLower(await SHA256.HashDataAsync(verify, cancellationToken).ConfigureAwait(false)) != hash)
                        throw new InvalidDataException("Upstream Qwen weight SHA-256 mismatch.");
                }
                File.Move(partial, target);
            }
            finally { if (File.Exists(partial)) File.Delete(partial); }
        }
        foreach (string file in new[] { "config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json", "generation_config.json" })
            await Fetch(file, null).ConfigureAwait(false);
        if (name == "1.7b") await Fetch("model.safetensors.index.json", null).ConfigureAwait(false);
        foreach (var file in pin.Item3) await Fetch(file.Item1, file.Item2).ConfigureAwait(false);
        File.WriteAllLines(Path.Combine(directory, "upstream.txt"), ["Qwen/" + pin.Item1, pin.Item2]);
        foreach (var file in pin.Item3)
        {
            using var stream = File.OpenRead(Path.Combine(directory, file.Item1));
            using var tensors = new SafeTensorFile(stream);
            foreach (var tensor in tensors.Tensors) Console.WriteLine($"{tensor.Key} {tensor.Value.DataType} [{string.Join(',', tensor.Value.Shape)}]");
        }
    }
}
