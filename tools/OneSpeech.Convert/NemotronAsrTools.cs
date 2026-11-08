using System.Buffers.Binary;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class NemotronAsrTools
{
    internal static void ProbeEncoder(string directory, string probes, int chunkFrames = 4)
    {
        var options = new ModelOptions { IntraOpThreads = 4, MaxCachedPlans = 2 };
        using var first = Session.Load(Path.Combine(directory, "encoder-part0.onemodel"), options);
        using var second = Session.Load(Path.Combine(directory, "encoder-part1.onemodel"), options);
        foreach (string file in Directory.EnumerateFiles(probes, "*.mel").Order(StringComparer.Ordinal))
        {
            float[] padded = Read(file); int physical = padded.Length / 128, frames = physical - 1, encodedFrames = (frames + 7) / 8;
            float[] mel = new float[128 * frames];
            for (int channel = 0; channel < 128; channel++) padded.AsSpan(channel * physical, frames).CopyTo(mel.AsSpan(channel * frames));
            float[] mask = new float[encodedFrames * encodedFrames];
            for (int q = 0; q < encodedFrames; q++)
            {
                int begin = q / chunkFrames * chunkFrames;
                for (int k = 0; k < encodedFrames; k++) if (k < begin - 56 || k >= begin + chunkFrames) mask[q * encodedFrames + k] = float.NegativeInfinity;
            }
            using var maskTensor = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, encodedFrames, encodedFrames));
            using var input = Tensor.FromSpan<float>(mel, Shape.Create(1, 128, frames));
            using var a = first.Run(new Dictionary<string, Tensor> { ["features"] = input, ["attention_mask"] = maskTensor });
            using var b = second.Run(new Dictionary<string, Tensor> { ["features"] = a["encoded"], ["attention_mask"] = maskTensor });
            var actual = b["encoded"].Data<float>(); float[] expected = Read(Path.ChangeExtension(file, ".encoded"));
            if (actual.Length != expected.Length) throw new InvalidDataException($"{Path.GetFileName(file)} length {actual.Length} != {expected.Length}");
            double sum = 0; float maximum = 0;
            for (int i = 0; i < actual.Length; i++) { float error = Math.Abs(actual[i] - expected[i]); sum += error; maximum = Math.Max(maximum, error); }
            Console.WriteLine($"{Path.GetFileName(file)} max={maximum:G9} mean={sum / actual.Length:G9}");
        }
    }
    private static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path); float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return values;
    }
}
