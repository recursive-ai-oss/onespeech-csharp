using OneSpeech.Catalog.Conversion;
using OneSpeech.Formats;
using OneSpeech.Moonshine;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class MoonshineFrontendProbe
{
    internal static void Run(string directory, string reference)
    {
        var c = MoonshineConfiguration.Load(directory);
        using var file = File.OpenRead(Path.Combine(directory, "model.safetensors"));
        using var weights = new SafeTensorFile(file);
        Console.WriteLine("log_k=" + weights.ReadFloat32("model.encoder.embedder.comp.log_k", CancellationToken.None)[0]);
        float[] data = Read("input_values.bin"), expected = Read("encoder-0.bin");
        using var input = Tensor.FromSpan<float>(data, Shape.Create(1, 1, data.Length));
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["audio"] = input };
        var rows = new List<(float Error, string Description)>();
        foreach (float epsilon in new[] { 1e-5f, 1e-6f, 1e-8f }) for (int norm = 0; norm < 6; norm++) for (int activation = 1; activation < 3; activation++)
        {
            var graph = new MoonshineGraphWriter(weights, compressionMode: 0, frontendActivation: activation, frontendEpsilon: epsilon, frontendNorm: norm);
            graph.FrontendProbe(c);
            string path = Path.Combine(directory, "frontend-probe.onemodel");
            graph.Save(path, CancellationToken.None);
            using var session = Session.Load(path, new ModelOptions { IntraOpThreads = 2 });
            using var result = session.Run(inputs);
            var actual = result["frontend"].Data<float>();
            if (actual.Length != expected.Length) throw new InvalidDataException("Frontend length mismatch.");
            float max = 0;
            double dot = 0, aa = 0, bb = 0;
            for (int i = 0; i < actual.Length; i++) { dot += actual[i] * expected[i]; aa += actual[i] * actual[i]; bb += expected[i] * expected[i]; }
            for (int i = 0; i < actual.Length; i++) max = Math.Max(max, Math.Abs(actual[i] - expected[i]));
            rows.Add((max, $"eps={epsilon:R} norm={norm} activation={activation} cosine={dot / Math.Sqrt(aa * bb):F6} scale={dot / aa:F6}"));
        }
        foreach (var row in rows.OrderBy(r => r.Error).Take(10)) Console.WriteLine($"{row.Description} maxAbs={row.Error:R}");
        File.Delete(Path.Combine(directory, "frontend-probe.onemodel"));
        float[] Read(string name)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(reference, name));
            float[] values = new float[bytes.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
            return values;
        }
    }
}
