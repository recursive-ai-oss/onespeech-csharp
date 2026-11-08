using System.Buffers.Binary;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;

internal static class ParakeetTools
{
    internal static void Vocabulary(string directory, string fixtures)
    {
        var vocabulary = new OneSpeech.FastConformer.ParakeetVocabulary(Path.Combine(directory, "tokenizer.json"));
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(fixtures));
        int count = 0, mismatches = 0;
        foreach (var row in document.RootElement.EnumerateArray())
        {
            int[] ids = row.GetProperty("ids").EnumerateArray().Select(item => item.GetInt32()).ToArray();
            string expected = row.GetProperty("text").GetString()!, actual = vocabulary.Decode(ids);
            if (expected != actual) { if (mismatches < 10) Console.WriteLine($"{string.Join(',', ids)} expected=[{expected}] actual=[{actual}]"); mismatches++; }
            count++;
        }
        Console.WriteLine($"Vocabulary decode cases {count}; mismatches {mismatches}");
    }
    internal static void Probe(string model, string probe)
    {
        using var first = Model.Load(Path.Combine(model, "encoder-part0.onemodel"), new ModelOptions { IntraOpThreads = 4 });
        using var second = Model.Load(Path.Combine(model, "encoder-part1.onemodel"), new ModelOptions { IntraOpThreads = 4 });
        foreach (string file in Directory.EnumerateFiles(probe, "*.mel").Order(StringComparer.Ordinal))
        {
            float[] input = Read(file);
            int padded = input.Length / 128;
            int frames = padded - 1;
            float[] trimmed = new float[128 * frames];
            for (int mel = 0; mel < 128; mel++) input.AsSpan(mel * padded, frames).CopyTo(trimmed.AsSpan(mel * frames));
            input = trimmed;
            float[] encoded = second.Run(first.Run(input, [1, 128, frames]), [1, 1024, (frames + 7) / 8]);
            float[] expected = Read(Path.ChangeExtension(file, ".encoded"));
            double sum = 0; float max = 0;
            if (encoded.Length != expected.Length) throw new InvalidDataException($"Length {encoded.Length} != {expected.Length}");
            for (int i = 0; i < encoded.Length; i++) { float error = Math.Abs(encoded[i] - expected[i]); max = Math.Max(max, error); sum += error; }
            Console.WriteLine($"{Path.GetFileName(file)} max={max:G8} mean={sum / encoded.Length:G8}");
        }
    }
    internal static void Batch(string directory, string corpus, string output, int begin, int count)
    {
        Directory.CreateDirectory(output);
        var info = new OneSpeech.ModelInfo { Id = "nvidia/parakeet-tdt-0.6b-v3", Revision = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", Source = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3", Tier = OneSpeech.ModelTier.Tier2, License = "CC-BY-4.0", Attribution = "NVIDIA, parakeet-tdt-0.6b-v3, CC-BY-4.0." };
        using var recognizer = OneSpeech.FastConformer.ParakeetRecognizer.Load(new(directory, info), threads: 4);
        foreach (string path in Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(200).Skip(begin).Take(count))
        {
            string id = Path.GetFileNameWithoutExtension(path), target = Path.Combine(output, id + ".json");
            if (File.Exists(target)) continue;
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var audio = new OneSpeech.AudioBuffer(clip.Samples, 16000);
            var tokens = recognizer.Decode(audio, CancellationToken.None);
            var transcript = recognizer.TranscribeDecoded(audio, tokens, "en", CancellationToken.None);
            using (var file = File.Create(target + ".partial"))
            using (var writer = new System.Text.Json.Utf8JsonWriter(file))
            {
                writer.WriteStartObject(); writer.WriteString("id", id); writer.WriteString("text", transcript.Text);
                writer.WriteStartArray("words");
                foreach (var word in transcript.Words)
                {
                    writer.WriteStartObject(); writer.WriteString("text", word.Text); writer.WriteNumber("start", word.Start!.Value.TotalSeconds); writer.WriteNumber("end", word.End!.Value.TotalSeconds); writer.WriteEndObject();
                }
                writer.WriteEndArray();
                {
                    writer.WriteStartArray("tokens"); foreach (var token in tokens) writer.WriteNumberValue(token.Id); writer.WriteEndArray();
                    writer.WriteStartArray("frames"); foreach (var token in tokens) writer.WriteNumberValue(token.StartFrame); writer.WriteEndArray();
                }
                writer.WriteEndObject();
            }
            File.Move(target + ".partial", target); Console.WriteLine(id + " " + transcript.Text);
        }
    }
    private static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path); float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return values;
    }
}
