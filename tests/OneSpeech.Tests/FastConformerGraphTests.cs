using RecursiveAI.OneModel.Graph;
using System.Buffers.Binary;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Formats;
using RecursiveAI.OneModel;
using Xunit;

namespace OneSpeech.Tests;

public sealed class FastConformerGraphTests
{
    [Fact]
    public void SmallRandomEncoderRunsAndSplitPreservesOutput()
    {
        var config = new FastConformerDimensions(MelBins: 16, Width: 8, Layers: 2, Heads: 2, Expansion: 2,
            ConvolutionKernel: 3, SubsamplingChannels: 4, PredictorWidth: 6, JointWidth: 5, Vocabulary: 4);
        var source = new SyntheticTensors(config);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-conformer-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var ct = TestContext.Current.CancellationToken;
        try
        {
            string all = Path.Combine(directory, "all.onemodel"), first = Path.Combine(directory, "first.onemodel"), second = Path.Combine(directory, "second.onemodel");
            new FastConformerGraphBuilder(source, config).Encoder(all, 0, 2, true, ct);
            new FastConformerGraphBuilder(source, config).Encoder(first, 0, 1, true, ct);
            new FastConformerGraphBuilder(source, config).Encoder(second, 1, 1, false, ct);
            using var complete = Model.Load(all, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var part0 = Model.Load(first, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var part1 = Model.Load(second, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            foreach (int frames in new[] { 64, 65 })
            {
                float[] input = Enumerable.Range(0, 16 * frames).Select(i => MathF.Sin(i * .1f)).ToArray();
                float[] actual = complete.Run(input, [1, 16, frames]);
                int encodedFrames = (frames + 7) / 8;
                float[] split = part1.Run(part0.Run(input, [1, 16, frames]), [1, 8, encodedFrames]);
                Assert.Equal(8 * encodedFrames, actual.Length);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                for (int i = 0; i < actual.Length; i++) Assert.Equal(actual[i], split[i], 5);
            }
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    public void SmallRandomPredictorAndJointRunWithExplicitState(int durationCount)
    {
        var config = new FastConformerDimensions(Width: 8, Heads: 2, PredictorWidth: 6, JointWidth: 5, Vocabulary: 4, DurationCount: durationCount);
        var source = new SyntheticTensors(config);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-transducer-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var ct = TestContext.Current.CancellationToken;
        try
        {
            string pred = Path.Combine(directory, "predictor.onemodel"), joint = Path.Combine(directory, "joint.onemodel");
            new FastConformerGraphBuilder(source, config).Predictor(pred, ct);
            new FastConformerGraphBuilder(source, config).Joint(joint, ct);
            using var predictor = Model.Load(pred, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            float[] state = predictor.Run(new float[30], [1, 30]);
            Assert.Equal(30, state.Length); Assert.All(state, value => Assert.True(float.IsFinite(value)));
            Assert.Equal(state, predictor.Run(new float[30], [1, 30]));
            using var joiner = Model.Load(joint, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            float[] logits = joiner.Run(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, state[0], state[1], state[2], state[3], state[4], state[5] }, [1, 14]);
            Assert.Equal(4 + durationCount, logits.Length); Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        }
        finally { Directory.Delete(directory, true); }
    }

    private sealed class SyntheticTensors : IFloatTensorSource
    {
        private readonly Dictionary<string, long[]> shapes = new(StringComparer.Ordinal);
        internal SyntheticTensors(FastConformerDimensions c)
        {
            void Add(string name, params long[] shape) => shapes.Add(name, shape);
            void Linear(string name, int input, int output, bool bias = false)
            { Add(name + ".weight", output, input); if (bias) Add(name + ".bias", output); }
            void Norm(string name) { Add(name + ".weight", c.Width); Add(name + ".bias", c.Width); }
            int d = c.Width, channels = c.SubsamplingChannels;
            Add("encoder.subsampling.0.weight", channels, 1, 3, 3); Add("encoder.subsampling.0.bias", channels);
            for (int stage = 1; stage < 3; stage++)
            {
                Add($"encoder.subsampling.{stage}.depthwise.weight", channels, 1, 3, 3); Add($"encoder.subsampling.{stage}.depthwise.bias", channels);
                Add($"encoder.subsampling.{stage}.pointwise.weight", channels, channels, 1, 1); Add($"encoder.subsampling.{stage}.pointwise.bias", channels);
            }
            Linear("encoder.subsampling.projection", channels * ((c.MelBins + 7) / 8), d, true);
            for (int layer = 0; layer < c.Layers; layer++)
            {
                string p = "encoder.layers." + layer;
                foreach (string ff in new[] { ".ff1", ".ff2" })
                { Norm(p + ff + ".norm"); Linear(p + ff + ".in", d, d * c.Expansion); Linear(p + ff + ".out", d * c.Expansion, d); }
                Norm(p + ".attention_norm"); Norm(p + ".conv_norm"); Norm(p + ".final_norm");
                foreach (string name in new[] { "q", "k", "v", "position", "out" }) Linear(p + ".attention." + name, d, d);
                Add(p + ".attention.u", c.Heads, d / c.Heads); Add(p + ".attention.v_bias", c.Heads, d / c.Heads);
                Add(p + ".conv.pointwise_in.weight", 2 * d, d, 1); Add(p + ".conv.depthwise.weight", d, 1, c.ConvolutionKernel); Add(p + ".conv.pointwise_out.weight", d, d, 1);
                foreach (string name in new[] { "weight", "bias", "mean", "variance" }) Add(p + ".conv.batch_norm." + name, d);
            }
            for (int layer = 0; layer < c.PredictorLayers; layer++)
            { Linear($"predictor.layers.{layer}.input", c.PredictorWidth, 4 * c.PredictorWidth, true); Linear($"predictor.layers.{layer}.hidden", c.PredictorWidth, 4 * c.PredictorWidth, true); }
            Linear("joint.encoder", d, c.JointWidth, true); Linear("joint.predictor", c.PredictorWidth, c.JointWidth, true); Linear("joint.output", c.JointWidth, c.Vocabulary + c.DurationCount, true);
        }
        public long[] GetShape(string name) => shapes[name];
        public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken)
        {
            int seed = 1729;
            foreach (char c in name) seed = unchecked(seed * 31 + c);
            var random = new Random(seed);
            byte[] buffer = new byte[4];
            long count = shapes[name].Aggregate(1L, (a, b) => a * b);
            for (long i = 0; i < count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float value = name.EndsWith(".variance", StringComparison.Ordinal) ? 1 : (float)(random.NextDouble() - .5) * .1f;
                if (name.Contains("norm.weight", StringComparison.Ordinal)) value += 1;
                BinaryPrimitives.WriteSingleLittleEndian(buffer, value); destination.Write(buffer);
            }
        }
    }
}
