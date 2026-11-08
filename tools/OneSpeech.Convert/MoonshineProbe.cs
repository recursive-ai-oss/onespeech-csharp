using System.Globalization;
using OneSpeech.Moonshine;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class MoonshineProbe
{
    internal static void Edges(string directory)
    {
        Directory.CreateDirectory(directory);
        foreach (var item in new[] { ("silence", 16000, 0), ("dc", 16000, 1), ("clipped", 16000, 2), ("offline-minimum", 895, 3), ("offline-odd", 912, 3), ("streaming-minimum", 80, 3), ("streaming-odd", 321, 3) })
        {
            using var file = File.Create(Path.Combine(directory, item.Item1 + ".pcm"));
            var bytes = new byte[item.Item2 * 4];
            for (int i = 0; i < item.Item2; i++)
            {
                float value = item.Item3 switch { 0 => 0, 1 => .125f, 2 => i % 2 == 0 ? 2f : -2f, _ => .25f * MathF.Sin(2 * MathF.PI * 440 * i / 16000) };
                System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), value);
            }
            file.Write(bytes);
        }
    }

    internal static void Run(string directory, string reference)
    {
        var c = MoonshineConfiguration.Load(directory);
        float[] audio = Read(Path.Combine(reference, File.Exists(Path.Combine(reference, "raw.bin")) ? "raw.bin" : "input_values.bin"));
        int frames = c.Streaming ? (audio.Length / 80 + 3) / 4 : (((audio.Length - 127) / 64 + 1 - 7) / 3 + 1 - 3) / 2 + 1;
        using var input = Tensor.FromSpan<float>(audio, Shape.Create(1, 1, audio.Length));
        using var cos = Rotary(frames, c.RotaryDimension, false);
        using var sin = Rotary(frames, c.RotaryDimension, true);
        using var encoder = Session.Load(Path.Combine(directory, "encoder.onemodel"), new ModelOptions { IntraOpThreads = 2 });
        using var local = LocalMask(frames, 4);
        using var causal = LocalMask(frames, 0);
        var encoderInputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["audio"] = input };
        if (c.Streaming) { encoderInputs["local_mask"] = local; encoderInputs["causal_mask"] = causal; }
        else { encoderInputs["cos"] = cos; encoderInputs["sin"] = sin; }
        using var encoded = encoder.Run(encoderInputs);
        if (File.Exists(Path.Combine(reference, "attention-0.bin")) && encoded.TryGetValue("attention-0", out var probability)) Compare("attention-0", probability, reference);
        for (int layer = 0; layer < c.EncoderLayers; layer++) if (encoded.TryGetValue("stage_" + layer, out var hidden)) Compare("encoder-" + layer, hidden, reference);
        Compare(File.Exists(Path.Combine(reference, "encoder-final.bin")) ? "encoder-final" : "encoder-" + c.EncoderLayers, encoded["encoder"], reference);
        using var decoder = Session.Load(Path.Combine(directory, "decoder.onemodel"), new ModelOptions { IntraOpThreads = 2 });
        using var tokens = Tensor.FromSpan<long>(new long[] { 1 }, Shape.Create(1, 1));
        using var dcos = Rotary(1, c.RotaryDimension, false);
        using var dsin = Rotary(1, c.RotaryDimension, true);
        using var mask = Tensor.Zeros(DType.F32, Shape.Create(1, 1, 1, 1));
        using var positions = Tensor.FromSpan<long>(Enumerable.Range(0, frames).Select(i => (long)i).ToArray(), Shape.Create(frames));
        var decoderInputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokens, ["encoder"] = encoded["encoder"], ["cos"] = dcos, ["sin"] = dsin, ["mask"] = mask };
        if (c.Streaming) decoderInputs["encoder_positions"] = positions;
        using var decoded = decoder.Run(decoderInputs);
        for (int layer = 0; layer < c.DecoderLayers; layer++) if (decoded.TryGetValue("stage_" + layer, out var hidden)) Compare("decoder-" + layer, hidden, reference);
        Compare("logits", decoded["logits"], reference);
    }
    private static Tensor LocalMask(int frames, int right)
    {
        var data = new float[frames * frames];
        for (int q = 0; q < frames; q++) for (int k = 0; k < frames; k++) if (k < q - 15 || k > q + Math.Max(0, right - 1)) data[q * frames + k] = float.NegativeInfinity;
        return Tensor.FromSpan<float>(data, Shape.Create(1, 1, frames, frames));
    }
    private static Tensor Rotary(int count, int dimension, bool sine)
    {
        var data = new float[count * dimension];
        for (int p = 0; p < count; p++) for (int d = 0; d < dimension; d++)
        {
            float angle = p * (float)Math.Pow(10000, -2d * (d / 2) / dimension);
            data[p * dimension + d] = sine ? MathF.Sin(angle) : MathF.Cos(angle);
        }
        return Tensor.FromSpan<float>(data, Shape.Create(1, 1, count, dimension));
    }
    private static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++) result[i] = System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return result;
    }
    private static void Compare(string name, Tensor actual, string directory)
    {
        float[] expected = Read(Path.Combine(directory, name + ".bin"));
        var data = actual.Data<float>();
        if (data.Length != expected.Length) throw new InvalidDataException($"{name} length {data.Length} != {expected.Length}");
        float maximum = 0, relative = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maximum = Math.Max(maximum, Math.Abs(data[i] - expected[i]));
            relative = Math.Max(relative, Math.Abs(data[i] - expected[i]) / Math.Max(1, Math.Abs(expected[i])));
        }
        Console.WriteLine($"{name} maxAbs={maximum.ToString("R", CultureInfo.InvariantCulture)} maxScaled={relative.ToString("R", CultureInfo.InvariantCulture)}");
    }
}
