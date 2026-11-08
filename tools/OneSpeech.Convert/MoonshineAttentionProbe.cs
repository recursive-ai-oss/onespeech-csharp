using OneSpeech.Formats;

internal static class MoonshineAttentionProbe
{
    internal static void Run(string directory, string reference)
    {
        using var input = File.OpenRead(Path.Combine(directory, "model.safetensors"));
        using var weights = new SafeTensorFile(input);
        var config = OneSpeech.Moonshine.MoonshineConfiguration.Load(directory);
        int width = config.Width, heads = config.EncoderHeads, dim = width / heads;
        float[] source = Read("encoder-0.bin"), expected = Read("attention-0.bin");
        int frames = source.Length / width;
        float[] qw = weights.ReadFloat32("model.encoder.layers.0.self_attn.q_proj.weight", CancellationToken.None);
        float[] kw = weights.ReadFloat32("model.encoder.layers.0.self_attn.k_proj.weight", CancellationToken.None);
        float[] gamma = weights.ReadFloat32("model.encoder.layers.0.input_layernorm." + (config.Streaming ? "gamma" : "weight"), CancellationToken.None);
        Console.WriteLine("gamma range " + gamma.Min() + " " + gamma.Max());
        var results = new List<(double Error, string Name)>();
        foreach (double epsilon in new[] { 1e-5, 1e-6, 1e-8 }) foreach (string norm in new[] { "ln-plus-one", "rms-plus-one" })
        {
            float[] q = new float[source.Length], k = new float[source.Length];
            for (int t = 0; t < frames; t++)
            {
                double mean = 0, variance = 0;
                if (norm is "ln" or "ln-plus-one" or "variance") for (int d = 0; d < width; d++) mean += source[t * width + d] / (double)width;
                for (int d = 0; d < width; d++) variance += Math.Pow(source[t * width + d] - mean, 2) / width;
                for (int row = 0; row < width; row++)
                {
                    double qs = 0, ks = 0;
                    for (int d = 0; d < width; d++)
                    {
                        double value = norm == "none" ? source[t * width + d] : (source[t * width + d] - (norm == "variance" ? 0 : mean)) / Math.Sqrt(variance + epsilon) * (gamma[d] + (norm.EndsWith("plus-one", StringComparison.Ordinal) ? 1 : 0));
                        qs += value * qw[row * width + d]; ks += value * kw[row * width + d];
                    }
                    q[t * width + row] = (float)qs; k[t * width + row] = (float)ks;
                }
            }
            foreach (int rotary in config.Streaming ? new[] { 0 } : new[] { 0, 32, 36 }) foreach (bool interleaved in new[] { false, true }) foreach (int left in new[] { 15, 16 }) foreach (int right in new[] { 3, 4 }) foreach (double divisor in new[] { Math.Sqrt(dim), Math.Sqrt(40), 1d })
            {
                var qr = Rotate(q, rotary, interleaved); var kr = Rotate(k, rotary, interleaved);
                double error = 0;
                var scores = new double[frames];
                for (int h = 0; h < heads; h++) for (int t = 0; t < frames; t++)
                {
                    double max = double.NegativeInfinity;
                    for (int s = 0; s < frames; s++)
                    {
                        double score = 0;
                        for (int d = 0; d < dim; d++) score += qr[t * width + h * dim + d] * kr[s * width + h * dim + d];
                        scores[s] = config.Streaming && (s < t - left || s > t + right) ? double.NegativeInfinity : score / divisor; max = Math.Max(max, scores[s]);
                    }
                    double sum = 0;
                    for (int s = 0; s < frames; s++) { scores[s] = Math.Exp(scores[s] - max); sum += scores[s]; }
                    for (int s = 0; s < frames; s++) error = Math.Max(error, Math.Abs(scores[s] / sum - expected[(h * frames + t) * frames + s]));
                }
                results.Add((error, $"epsilon={epsilon} left={left} right={right} norm={norm} rotary={rotary} interleaved={interleaved} divisor={divisor}"));
            }
        }
        foreach (var r in results.OrderBy(r => r.Error).Take(8)) Console.WriteLine($"{r.Name} maxAbs={r.Error:R}");
        float[] Read(string name)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(reference, name));
            float[] values = new float[bytes.Length / 4];
            for (int i = 0; i < values.Length; i++) values[i] = System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
            return values;
        }
        float[] Rotate(float[] values, int rotary, bool interleaved)
        {
            var result = values.ToArray();
            for (int t = 0; t < frames; t++) for (int h = 0; h < heads; h++) for (int d = 0; d < rotary / 2; d++)
            {
                int a = t * width + h * dim + (interleaved ? d * 2 : d);
                int b = t * width + h * dim + (interleaved ? d * 2 + 1 : d + rotary / 2);
                double angle = t * Math.Pow(10000, -2d * d / rotary);
                result[a] = (float)(values[a] * Math.Cos(angle) - values[b] * Math.Sin(angle));
                result[b] = (float)(values[b] * Math.Cos(angle) + values[a] * Math.Sin(angle));
            }
            return result;
        }
    }
}
