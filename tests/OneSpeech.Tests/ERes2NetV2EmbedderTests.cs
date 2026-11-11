using OneSpeech.Embedders;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ERes2NetV2EmbedderTests
{
    private static readonly ModelInfo Info = new() { Id = "algebraic", Source = "https://example.org", Revision = "none", License = "MIT", Attribution = "Algebraic test graph.", Tier = ModelTier.Tier1 };

    [Fact]
    public void CallerBufferLifetimeCancellationAndValidation()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onemodel");
        var graph = new SpeakerGraph();
        float[] basis = new float[192]; basis[0] = 1;
        graph.Save(path, graph.Float("basis", [1, 192], basis), 192);
        try
        {
            using var embedder = new ERes2NetV2Embedder(path, Info);
            float[] output = new float[192];
            var valid = new AudioBuffer(new float[16000], 16000);
            embedder.Embed(valid, output, CancellationToken.None);
            Assert.Equal(1, output[0]); Assert.Equal(0, output[1]);
            Assert.Throws<ArgumentException>(() => embedder.Embed(new AudioBuffer(new float[3000], 16000), output, CancellationToken.None));
            Assert.Throws<ArgumentException>(() => embedder.Embed(new AudioBuffer(new float[16000], 8000), output, CancellationToken.None));
            Assert.Throws<ArgumentException>(() => embedder.Embed(valid, new float[512], CancellationToken.None));
            Assert.Throws<OperationCanceledException>(() => embedder.Embed(valid, output, new CancellationToken(true)));
            embedder.Dispose(); embedder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => embedder.Embed(valid, output, CancellationToken.None));
        }
        finally { File.Delete(path); }
    }

    // A miniature of the published topology (same tensor names and wiring, 2-4 channels per stage, deterministic
    // pseudo-weights) proves the converter builds a graph that runs at the minimum and at a typical length.
    [Theory]
    [InlineData(3200)]
    [InlineData(24000)]
    public void ConvertedMiniatureRunsAndProducesFiniteEmbeddings(int samples)
    {
        var tensors = Miniature();
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onemodel");
        try
        {
            new ERes2NetV2Converter(tensors).Save(path);
            using var embedder = new ERes2NetV2Embedder(path, Info);
            float[] pcm = new float[samples];
            for (int i = 0; i < samples; i++) pcm[i] = 0.3f * MathF.Sin(i * 0.05f) + 0.1f * MathF.Sin(i * 0.31f);
            float[] output = new float[192];
            embedder.Embed(new AudioBuffer(pcm, 16000), output, CancellationToken.None);
            Assert.All(output, v => Assert.True(float.IsFinite(v)));
            Assert.Contains(output, v => v != 0);
        }
        finally { File.Delete(path); }
    }

    private static Dictionary<string, TensorCheckpoint.Tensor> Miniature()
    {
        var t = new Dictionary<string, TensorCheckpoint.Tensor>(StringComparer.Ordinal);
        uint state = 12345;
        float Next() { state = state * 1664525u + 1013904223u; return ((state >> 8) / (float)(1 << 24) - 0.5f) * 0.4f; }
        void Add(string name, params int[] shape) { float[] v = new float[shape.Aggregate(1, (a, b) => a * b)]; for (int i = 0; i < v.Length; i++) v[i] = Next(); t[name] = new(shape, v); }
        void Conv(string name, int o, int i, int k, bool bias = false) { Add(name + ".weight", o, i, k, k); if (bias) Add(name + ".bias", o); }
        void Bn(string name, int c)
        {
            t[name + ".weight"] = new([c], Enumerable.Repeat(1f, c).ToArray()); t[name + ".bias"] = new([c], new float[c]);
            t[name + ".running_mean"] = new([c], new float[c]); t[name + ".running_var"] = new([c], Enumerable.Repeat(1f, c).ToArray());
        }
        void Fuse(string p, int c) { Conv(p + ".local_att.0", c / 4 + 1, 2 * c, 1, true); Bn(p + ".local_att.1", c / 4 + 1); Conv(p + ".local_att.3", c, c / 4 + 1, 1, true); Bn(p + ".local_att.4", c); }
        int stem = 4;
        Conv("conv1", stem, 1, 3); Bn("bn1", stem);
        int[] counts = [3, 4, 6, 3], width = [2, 2, 2, 2], outputs = [4, 4, 4, 4];
        int previous = stem;
        for (int layer = 1; layer <= 4; layer++)
            for (int b = 0; b < counts[layer - 1]; b++)
            {
                string p = $"layer{layer}.{b}";
                int w = width[layer - 1], o = outputs[layer - 1];
                Conv(p + ".conv1", 2 * w, previous, 1); Bn(p + ".bn1", 2 * w);
                Conv(p + ".convs.0", w, w, 3); Conv(p + ".convs.1", w, w, 3); Bn(p + ".bns.0", w); Bn(p + ".bns.1", w);
                if (layer >= 3) Fuse(p + ".fuse_models.0", w);
                Conv(p + ".conv3", o, 2 * w, 1); Bn(p + ".bn3", o);
                if (previous != o || (layer > 1 && b == 0)) { Conv(p + ".shortcut.0", o, previous, 1); Bn(p + ".shortcut.1", o); }
                previous = o;
            }
        Conv("layer3_ds", 4, 4, 3); Fuse("fuse34", 4);
        Add("seg_1.weight", 192, 4 * 10 * 2); Add("seg_1.bias", 192);
        return t;
    }
}
