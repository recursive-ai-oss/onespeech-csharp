using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Formats;
using OneSpeech.FastConformer;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class NemotronAsrTools
{
    internal static void ProbeEncoder(string directory, string probes, int chunkFrames = 4, string? transformerProbes = null)
    {
        var options = new ModelOptions { IntraOpThreads = 4, MaxCachedPlans = 2 };
        using var first = Session.Load(Path.Combine(directory, "encoder-part0.onemodel"), options);
        using var second = Session.Load(Path.Combine(directory, "encoder-part1.onemodel"), options);
        foreach (string file in Directory.EnumerateFiles(probes, "*.mel").Order(StringComparer.Ordinal))
        {
            float[] physicalMel = Read(file); int physicalFrames = physicalMel.Length / 128;
            int frames = checked((int)(new FileInfo(Path.ChangeExtension(file, ".pcm")).Length / 4 / 160));
            int encodedFrames = ((frames / 2 + 1) / 2 + 1) / 2 + 1;
            int validEncoded = encodedFrames;
            float[] mel = new float[128 * frames];
            for (int c = 0; c < 128; c++) physicalMel.AsSpan(c * physicalFrames, frames).CopyTo(mel.AsSpan(c * frames));
            float[] mask = new float[encodedFrames * encodedFrames];
            for (int q = 0; q < encodedFrames; q++)
            {
                int begin = q / chunkFrames * chunkFrames;
                for (int k = 0; k < encodedFrames; k++) if (k < begin - 56 || k >= begin + chunkFrames || k >= validEncoded) mask[q * encodedFrames + k] = float.NegativeInfinity;
            }
            using var maskTensor = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, encodedFrames, encodedFrames));
            using var input = Tensor.FromSpan<float>(mel, Shape.Create(1, 128, frames));
            using var a = first.Run(new Dictionary<string, Tensor> { ["features"] = input, ["attention_mask"] = maskTensor });
            using var b = second.Run(new Dictionary<string, Tensor> { ["features"] = a["encoded"], ["attention_mask"] = maskTensor });
            var actual = b["encoded"].Data<float>(); float[] physicalExpected = Read(Path.ChangeExtension(file, ".encoded"));
            int physicalEncoded = physicalExpected.Length / 1024;
            float[] expected = new float[1024 * encodedFrames];
            for (int c = 0; c < 1024; c++) physicalExpected.AsSpan(c * physicalEncoded, encodedFrames).CopyTo(expected.AsSpan(c * encodedFrames));
            if (actual.Length != expected.Length) throw new InvalidDataException($"{Path.GetFileName(file)} length {actual.Length} != {expected.Length}");
            double sum = 0; float maximum = 0;
            for (int i = 0; i < actual.Length; i++) { float error = Math.Abs(actual[i] - expected[i]); sum += error; maximum = Math.Max(maximum, error); }
            Console.WriteLine($"{Path.GetFileName(file)} max={maximum:G9} mean={sum / actual.Length:G9}");
            if (maximum > 1e-5f) throw new InvalidDataException("Encoder parity exceeded 1e-5 maximum absolute error.");
            if (transformerProbes != null)
            {
                string root = Path.Combine(transformerProbes, Path.GetFileNameWithoutExtension(file));
                var transformerMel = Read(root + ".features");
                var transformerHidden = Read(root + ".hidden24");
                Compare("mel", mel, transformerMel.AsSpan(0, 128 * frames), 128, frames);
                Compare("native-vs-hf", expected, transformerHidden.AsSpan(0, 1024 * encodedFrames), 1024, encodedFrames);
                Compare("ours-vs-hf", actual, transformerHidden.AsSpan(0, 1024 * encodedFrames), 1024, encodedFrames);
                for (int layer = 0; layer <= 24; layer++)
                {
                    var outputs = layer <= 12 ? a : b;
                    string key = "hidden" + layer;
                    if (outputs.TryGetValue(key, out var hidden))
                    {
                        float[] reference = Read(root + ".hidden" + layer);
                        double total = 0; float max = 0;
                        var values = hidden.Data<float>();
                        for (int i = 0; i < values.Length; i++) { float error = Math.Abs(values[i] - reference[i]); total += error; max = Math.Max(max, error); }
                        Console.WriteLine($"layer={layer} max={max:G9} mean={total / values.Length:G9}");
                    }
                }
            }
        }
    }
    internal static void ProbeDecoder(string directory, string probes, string transformerProbes)
    {
        var options = new ModelOptions { IntraOpThreads = 2 };
        using var prompt = Session.Load(Path.Combine(directory, "prompt.onemodel"), options);
        using var predictor = Session.Load(Path.Combine(directory, "predictor.onemodel"), options);
        using var joint = Session.Load(Path.Combine(directory, "joint.onemodel"), options);
        float[] embeddings = Read(Path.Combine(directory, "embedding.f32"));
        float[] Run(Session session, string inputName, float[] values, params int[] shape)
        {
            using var input = Tensor.FromSpan<float>(values, Shape.Create(shape));
            using var output = session.Run(new Dictionary<string, Tensor> { [inputName] = input });
            return output[inputName == "prompt_input" ? "conditioned" : inputName == "predictor_state" ? "next_predictor_state" : "logits"].Data<float>().ToArray();
        }
        foreach (string file in Directory.EnumerateFiles(probes, "*.encoded").Order(StringComparer.Ordinal))
        {
            float[] encoder = Read(file); int physical = encoder.Length / 1024;
            int melLength = checked((int)(new FileInfo(Path.ChangeExtension(file, ".pcm")).Length / 4 / 160));
            int frames = ((melLength / 2 + 1) / 2 + 1) / 2 + 1;
            float[] promptInput = new float[1152 * frames];
            for (int channel = 0; channel < 1024; channel++) encoder.AsSpan(channel * physical, frames).CopyTo(promptInput.AsSpan(channel * frames));
            promptInput.AsSpan(1024 * frames, frames).Fill(1); // en-US prompt slot zero.
            float[] conditioned = Run(prompt, "prompt_input", promptInput, 1, 1152, frames);
            float[] state = new float[3200], jointInput = new float[1664];
            embeddings.AsSpan(13087 * 640, 640).CopyTo(state);
            state = Run(predictor, "predictor_state", state, 1, 3200);
            List<int> tokens = [];
            for (int frame = 0; frame < frames; frame++)
            {
                for (int c = 0; c < 1024; c++) jointInput[c] = conditioned[c * frames + frame];
                for (int symbol = 0; symbol < 10; symbol++)
                {
                    state.AsSpan(0, 640).CopyTo(jointInput.AsSpan(1024));
                    float[] logits = Run(joint, "joint_input", jointInput, 1, 1664);
                    int best = 0;
                    for (int id = 1; id < logits.Length; id++) if (logits[id] > logits[best]) best = id;
                    if (best == 13087) break;
                    tokens.Add(best); embeddings.AsSpan(best * 640, 640).CopyTo(state);
                    state = Run(predictor, "predictor_state", state, 1, 3200);
                }
            }
            using var expectedJson = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(transformerProbes, Path.GetFileNameWithoutExtension(file) + ".json")));
            int[] expected = expectedJson.RootElement.GetProperty("tokens")[0].EnumerateArray().Select(x => x.GetInt32()).Where(x => x != 13087).ToArray();
            Console.WriteLine($"{Path.GetFileName(file)} tokens={tokens.Count}/{expected.Length} equal={tokens.SequenceEqual(expected)}");
            if (!tokens.SequenceEqual(expected)) throw new InvalidDataException("Decoder token mismatch: actual=" + string.Join(',', tokens) + " expected=" + string.Join(',', expected));
        }
    }
    internal static void ProbeCaches(string directory, string probes)
    {
        var options = new ModelOptions { IntraOpThreads = 2, MaxCachedPlans = 3 };
        using var first = Session.Load(Path.Combine(directory, "encoder-part0.onemodel"), options);
        using var second = Session.Load(Path.Combine(directory, "encoder-part1.onemodel"), options);
        using var stepFirst = Session.Load(Path.Combine(directory, "encoder-step0.onemodel"), options);
        using var stepSecond = Session.Load(Path.Combine(directory, "encoder-step1.onemodel"), options);
        using var tailFirst = Session.Load(Path.Combine(directory, "encoder-tail0.onemodel"), options);
        string file = Directory.EnumerateFiles(probes, "*.mel").Order(StringComparer.Ordinal).Skip(1).First();
        float[] physicalMel = Read(file); int physicalFrames = physicalMel.Length / 128;
        int frames = checked((int)(new FileInfo(Path.ChangeExtension(file, ".pcm")).Length / 4 / 160));
        int encodedFrames = ((frames / 2 + 1) / 2 + 1) / 2 + 1;
        float[] mel = new float[128 * frames];
        for (int c = 0; c < 128; c++) physicalMel.AsSpan(c * physicalFrames, frames).CopyTo(mel.AsSpan(c * frames));
        using var input = Tensor.FromSpan<float>(mel, Shape.Create(1, 128, frames));
        foreach (int chunk in new[] { 1, 4, 7, 14 })
        {
            float[] mask = new float[encodedFrames * encodedFrames];
            for (int q = 0; q < encodedFrames; q++) for (int k = 0; k < encodedFrames; k++) if (k < q / chunk * chunk - 56 || k >= q / chunk * chunk + chunk) mask[q * encodedFrames + k] = float.NegativeInfinity;
            using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, encodedFrames, encodedFrames));
            using var a = first.Run(new Dictionary<string, Tensor> { ["features"] = input, ["attention_mask"] = maskInput });
            using var b = second.Run(new Dictionary<string, Tensor> { ["features"] = a["encoded"], ["attention_mask"] = maskInput });
            var expected = b["encoded"].Data<float>().ToArray();
            float[] actual = new float[expected.Length], frame = new float[128]; int used = 0;
            void Emit(ReadOnlySpan<float> values, int valid, CancellationToken ct)
            {
                ct.ThrowIfCancellationRequested();
                for (int c = 0; c < 1024; c++) values.Slice(c * chunk, valid).CopyTo(actual.AsSpan(c * encodedFrames + used));
                used += valid;
            }
            using var state = new NemotronAsrEncoderState(stepFirst, stepSecond, tailFirst, NemotronAsrConverter.Dimensions, chunk, Emit, CancellationToken.None);
            for (int t = 0; t < frames; t++) { for (int c = 0; c < 128; c++) frame[c] = mel[c * frames + t]; state.Append(frame, CancellationToken.None); }
            state.Complete(CancellationToken.None);
            float max = 0; double total = 0;
            for (int i = 0; i < actual.Length; i++) { float error = Math.Abs(actual[i] - expected[i]); total += error; max = Math.Max(max, error); }
            Console.WriteLine($"cache chunk={chunk} frames={used}/{encodedFrames} max={max:G9} mean={total / actual.Length:G9}");
            if (used != encodedFrames || max > 1e-5f) throw new InvalidDataException("Cached encoder parity exceeded 1e-5 maximum absolute error.");
        }
    }
    private static void Compare(string name, ReadOnlySpan<float> channelMajor, ReadOnlySpan<float> timeMajor, int channels, int frames)
    {
        if (channelMajor.Length != timeMajor.Length) throw new InvalidDataException("Mismatched probe shapes.");
        double sum = 0; float maximum = 0;
        for (int c = 0; c < channels; c++) for (int t = 0; t < frames; t++)
        {
            float error = Math.Abs(channelMajor[c * frames + t] - timeMajor[t * channels + c]);
            sum += error; maximum = Math.Max(maximum, error);
        }
        Console.WriteLine($"{name} max={maximum:G9} mean={sum / channelMajor.Length:G9}");
    }
    private static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path); float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return values;
    }
}
