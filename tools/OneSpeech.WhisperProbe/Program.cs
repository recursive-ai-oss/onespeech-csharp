using System.Buffers.Binary;
using System.Globalization;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Whisper;
using System.Text.Json;

if (args.Length == 3 && args[0] == "long-features")
{
    var audio = new AudioBuffer(ReadFloats(args[1]), 16000); Directory.CreateDirectory(args[2]);
    foreach (int bins in new[] { 80, 128 })
    {
        var extractor = new WhisperLogMelSpectrogram(bins); float floor = float.NegativeInfinity;
        for (int offset = 0; offset < audio.FrameCount; offset += 480000) floor = MathF.Max(floor, extractor.ExtractContext(audio, offset, float.NegativeInfinity, CancellationToken.None).Max() - 2);
        for (int offset = 0; offset < audio.FrameCount; offset += 480000) WriteFloats(Path.Combine(args[2], $"{offset}.{bins}.f32"), extractor.ExtractContext(audio, offset, floor, CancellationToken.None));
    }
    return 0;
}
if (args.Length == 8 && args[0] == "realign")
{
    string[] ids = File.ReadAllLines(args[3]); Directory.CreateDirectory(args[5]);
    using var recognizer = WhisperRecognizer.Load(args[1], new WhisperOptions { Threads = 8 });
    int begin = int.Parse(args[6], CultureInfo.InvariantCulture), count = int.Parse(args[7], CultureInfo.InvariantCulture);
    for (int i = begin; i < Math.Min(ids.Length, begin + count); i++)
    {
        string path = Path.Combine(args[5], ids[i] + ".json"); if (File.Exists(path)) continue;
        Transcript original = Transcript.FromJson(File.ReadAllText(Path.Combine(args[4], ids[i] + ".json")));
        Transcript aligned = recognizer.Transcribe(new AudioBuffer(ReadFloats(Path.Combine(args[2], ids[i] + ".pcm")), 16000), original.Language, CancellationToken.None);
        File.WriteAllText(path + ".part", aligned.ToJson()); File.Move(path + ".part", path); Console.WriteLine(ids[i]);
    }
    return 0;
}
if (args.Length == 4 && args[0] == "greedy")
{
    using var recognizer = WhisperRecognizer.Load(args[1], new WhisperOptions { Threads = 4, Temperatures = [0], WordTimestamps = false });
    int[] tokens = recognizer.DecodeProbe(ReadFloats(args[2]), "en", CancellationToken.None);
    using var output = File.Create(args[3]); using var json = new Utf8JsonWriter(output);
    json.WriteStartArray(); foreach (int token in tokens) json.WriteNumberValue(token); json.WriteEndArray();
    return 0;
}
if (args.Length == 4 && (args[0] == "norm" || args[0] == "block"))
{
    var config = WhisperConfiguration.Load(args[1]);
    using var input = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(ReadFloats(args[2]), RecursiveAI.OneModel.Tensors.Shape.Create(1, 1500, config.Width));
    using var session = RecursiveAI.OneModel.Graph.Session.Load(Path.Combine(args[1], args[0] == "norm" ? "norm-probe.onemodel" : "block-probe.onemodel"), new RecursiveAI.OneModel.Graph.ModelOptions { IntraOpThreads = 4 });
    using var output = session.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor>(StringComparer.Ordinal) { ["hidden"] = input });
    Directory.CreateDirectory(args[3]); WriteFloats(Path.Combine(args[3], args[0] == "norm" ? "encoder.f32" : "block" + (config.EncoderLayers == 12 ? 6 : 20) + ".f32"), output["encoder"].Data<float>());
    return 0;
}
if (args.Length == 4 && args[0] == "stages")
{
    Directory.CreateDirectory(args[3]);
    var config = WhisperConfiguration.Load(args[1]);
    using var input = RecursiveAI.OneModel.Tensors.Tensor.FromSpan<float>(ReadFloats(args[2]), RecursiveAI.OneModel.Tensors.Shape.Create(1, config.MelBins, 3000));
    using var first = File.Exists(Path.Combine(args[1], "encoder-part0.debug.onemodel")) ? RecursiveAI.OneModel.Graph.Session.Load(Path.Combine(args[1], "encoder-part0.debug.onemodel"), new RecursiveAI.OneModel.Graph.ModelOptions { IntraOpThreads = 4 }) : null;
    using var intermediate = first?.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor>(StringComparer.Ordinal) { ["mel"] = input });
    if (intermediate != null)
        foreach (string name in intermediate.Names) if (name != "encoder") WriteFloats(Path.Combine(args[3], name + ".f32"), intermediate[name].Data<float>());
    using var final = RecursiveAI.OneModel.Graph.Session.Load(Path.Combine(args[1], "encoder.debug.onemodel"), new RecursiveAI.OneModel.Graph.ModelOptions { IntraOpThreads = 4 });
    using var output = final.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor>(StringComparer.Ordinal) { [intermediate == null ? "mel" : "hidden"] = intermediate == null ? input : intermediate["encoder"] });
    foreach (string name in output.Names) if (!name.StartsWith("cross", StringComparison.Ordinal)) WriteFloats(Path.Combine(args[3], name + ".f32"), output[name].Data<float>());
    return 0;
}
if (args.Length == 5 && args[0] == "export")
{
    ExportHypotheses.Run(args[1], args[2], args[3], args[4]);
    return 0;
}
if (args.Length == 8 && args[0] == "recognize")
{
    string[] ids = File.ReadAllLines(args[3]);
    Directory.CreateDirectory(args[4]);
    int begin = int.Parse(args[5], CultureInfo.InvariantCulture), count = int.Parse(args[6], CultureInfo.InvariantCulture);
    bool words = args[7] == "words";
    using var recognizer = WhisperRecognizer.Load(args[1], new WhisperOptions { Threads = 8, WordTimestamps = words });
    for (int i = begin; i < Math.Min(ids.Length, begin + count); i++)
    {
        string path = Path.Combine(args[4], ids[i] + ".json");
        if (File.Exists(path)) continue;
        var audio = new AudioBuffer(ReadFloats(Path.Combine(args[2], ids[i] + ".pcm")), 16000);
        Transcript transcript = recognizer.Transcribe(audio, "en", CancellationToken.None);
        File.WriteAllText(path + ".part", transcript.ToJson()); File.Move(path + ".part", path);
        if (i < 50 && words) File.WriteAllText(Path.Combine(args[4], ids[i] + ".language"), recognizer.DetectLanguage(audio, CancellationToken.None));
        Console.WriteLine(ids[i]);
    }
    return 0;
}
if (args.Length == 3 && args[0] == "transcribe")
{
    using var recognizer = WhisperRecognizer.Load(args[1], new WhisperOptions { Threads = 4 });
    Transcript transcript = recognizer.Transcribe(new AudioBuffer(ReadFloats(args[2]), 16000), "en", CancellationToken.None);
    Console.WriteLine(transcript.ToJson());
    return 0;
}
if (args.Length == 4 && args[0] == "tokenizer")
{
    var tokenizer = new WhisperTokenizer(args[1]);
    using JsonDocument cases = JsonDocument.Parse(File.ReadAllText(args[2]));
    using FileStream file = File.Create(args[3]);
    using var json = new Utf8JsonWriter(file);
    json.WriteStartArray();
    foreach (var item in cases.RootElement.EnumerateArray())
    {
        string text = item.GetString()!;
        int[] tokens = tokenizer.Encode(text, allowSpecialTokens: true);
        json.WriteStartObject(); json.WriteString("text", text); json.WriteString("decoded", tokenizer.Decode(tokens, includeSpecialTokens: true));
        json.WriteStartArray("ids"); foreach (int token in tokens) json.WriteNumberValue(token); json.WriteEndArray(); json.WriteEndObject();
    }
    json.WriteEndArray();
    return 0;
}
if (args.Length == 3 && args[0] == "compare")
{
    Console.WriteLine("file,max_abs,scaled_error,mean_abs");
    foreach (string path in Directory.GetFiles(args[1], "*.f32").Order(StringComparer.Ordinal))
    {
        float[] actual = ReadFloats(path), reference = ReadFloats(Path.Combine(args[2], Path.GetFileName(path)));
        if (actual.Length != reference.Length) throw new InvalidDataException("Shape mismatch.");
        int maxIndex = 0; double max = 0, scale = 0, sum = 0;
        for (int i = 0; i < actual.Length; i++) { double error = Math.Abs(actual[i] - reference[i]); if (error > max) { max = error; maxIndex = i; } scale = Math.Max(scale, Math.Abs(reference[i])); sum += error; }
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{Path.GetFileName(path)},{max:G9},{max / scale:G9},{sum / actual.Length:G9}"));
        Console.Error.WriteLine($"maximum index {maxIndex}, values {actual[maxIndex]:G9} vs {reference[maxIndex]:G9}");
    }
    return 0;
}
if (args.Length == 4 && args[0] == "graph")
{
    using var model = new WhisperModelSession(args[1], 4, CancellationToken.None);
    float[] samples = ReadFloats(args[2]);
    float[] features = args[2].EndsWith(".f32", StringComparison.Ordinal) ? samples : new WhisperLogMelSpectrogram(model.Configuration.MelBins).Extract(new AudioBuffer(samples, 16000));
    using var encoded = model.Encode(features, CancellationToken.None);
    WriteFloats(args[3] + ".encoder.f32", encoded["encoder"].Data<float>());
    using var generation = JsonDocument.Parse(File.ReadAllText(Path.Combine(args[1], "generation_config.json")));
    var g = generation.RootElement;
    int[] prefix = [g.GetProperty("decoder_start_token_id").GetInt32(), g.GetProperty("lang_to_id").GetProperty("<|en|>").GetInt32(), g.GetProperty("task_to_id").GetProperty("transcribe").GetInt32()];
    using var decoded = model.Decode(encoded, prefix, null, 0, CancellationToken.None);
    WriteFloats(args[3] + ".logits.f32", decoded["logits"].Data<float>());
    return 0;
}


if (args.Length != 1) { Console.Error.WriteLine("Usage: OneSpeech.WhisperProbe <oracle-output-directory>"); return 2; }
Console.WriteLine("clip,bins,reference,max_abs,mean_abs");
foreach (string path in Directory.GetFiles(args[0], "*.pcm").Order(StringComparer.Ordinal))
{
    float[] audio = ReadFloats(path);
    foreach (int bins in new[] { 80, 128 })
    {
        float[] actual = new WhisperLogMelSpectrogram(bins).Extract(new AudioBuffer(audio, 16000));
        foreach (string reference in new[] { "whisper", "transformers" })
        {
            string expectedPath = Path.Combine(args[0], $"{Path.GetFileNameWithoutExtension(path)}.{bins}.{reference}.f32");
            float[] expected = ReadFloats(expectedPath);
            if (actual.Length != expected.Length) throw new InvalidDataException("Feature shape mismatch.");
            double max = 0, sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double error = Math.Abs(actual[i] - expected[i]);
                if (!double.IsFinite(error)) throw new InvalidDataException("Nonfinite feature error.");
                max = Math.Max(max, error);
                sum += error;
            }
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{Path.GetFileNameWithoutExtension(path)},{bins},{reference},{max:G9},{sum / actual.Length:G9}"));
        }
    }
}
return 0;

static float[] ReadFloats(string path)
{
    using FileStream stream = File.OpenRead(path);
    if (stream.Length % 4 != 0 || stream.Length > int.MaxValue) throw new InvalidDataException("Invalid float file length.");
    byte[] bytes = new byte[stream.Length];
    stream.ReadExactly(bytes);
    float[] result = new float[bytes.Length / 4];
    for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
    return result;
}

static void WriteFloats(string path, ReadOnlySpan<float> values)
{
    using var file = File.Create(path);
    byte[] buffer = new byte[65536];
    for (int offset = 0; offset < values.Length;)
    {
        int count = Math.Min(values.Length - offset, buffer.Length / 4);
        for (int i = 0; i < count; i++) BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(i * 4), values[offset + i]);
        file.Write(buffer, 0, count * 4); offset += count;
    }
}
