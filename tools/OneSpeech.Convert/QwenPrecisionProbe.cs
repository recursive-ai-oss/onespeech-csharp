using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Qwen3;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class QwenPrecisionProbe
{
    internal static void Run(string directory)
    {
        const int width = 1024;
        Directory.CreateDirectory(directory);
        foreach (string mode in new[] { "f32", "round-f32-weights", "round-bf16-weights" })
        {
            string root = Path.Combine(directory, mode); Directory.CreateDirectory(root);
            bool compact = mode == "round-bf16-weights";
            string header = "{\"thinker.lm_head.weight\":{\"dtype\":\"" + (compact ? "BF16" : "F32") + "\",\"shape\":[1024,1024],\"data_offsets\":[0," + width * width * (compact ? 2 : 4) + "]}}";
            using (var file = File.Create(Path.Combine(root, "model.safetensors")))
            using (var writer = new BinaryWriter(file, Encoding.UTF8, leaveOpen: true))
            {
                byte[] json = Encoding.UTF8.GetBytes(header); writer.Write((ulong)json.Length); writer.Write(json);
                uint state = 1729;
                for (int i = 0; i < width * width; i++)
                {
                    state = unchecked(state * 1664525 + 1013904223);
                    float value = ((int)(state >> 24) - 128) / 128f;
                    if (compact) writer.Write((ushort)(BitConverter.SingleToUInt32Bits(value) >> 16)); else writer.Write(value);
                }
            }
            using (var weights = new QwenWeights(root))
            {
                var graph = new QwenGraphWriter(weights, mode != "f32");
                graph.LanguageHead(new QwenConfiguration(width, 16, 1, 1, width, 8, 8, 128, 1, width, 0));
                graph.Save(Path.Combine(root, "head.onemodel"), CancellationToken.None);
            }
            using var session = Session.Load(Path.Combine(root, "head.onemodel"), new ModelOptions { IntraOpThreads = 1, MaxCachedPlans = 2 });
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var watch = Stopwatch.StartNew();
            byte[] bytes = new byte[width * 4];
            for (int step = 0; step < 32; step++)
            {
                int sequence = step % 8 + 1;
                float[] values = new float[sequence * width];
                for (int i = 0; i < values.Length; i++) values[i] = (i % 32 - 16) / 32f;
                using var input = Tensor.FromSpan<float>(values, Shape.Create(1, sequence, width));
                using var result = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = input });
                var logits = result["logits"].Data<float>();
                for (int i = 0; i < logits.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), logits[i]);
                hash.AppendData(bytes);
            }
            watch.Stop();
            Console.WriteLine(mode + " iterations=32 threads=1 elapsed_ms=" + watch.Elapsed.TotalMilliseconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + " output_sha256=" + Convert.ToHexStringLower(hash.GetHashAndReset()));
        }
    }
}
