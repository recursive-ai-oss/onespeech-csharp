using OneSpeech.Catalog.Conversion;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using Xunit;

namespace OneSpeech.Tests;

public sealed class NemotronGraphTests
{
    [Fact]
    public void ChunkMaskAndCausalConvolutionPreventFutureChunkLeakage()
    {
        var dimensions = new FastConformerDimensions(MelBins: 16, Width: 8, Layers: 2, Heads: 2, Expansion: 2, ConvolutionKernel: 3, SubsamplingChannels: 32);
        var source = new FastConformerGraphTests.SyntheticTensors(dimensions, causal: true);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-causal-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var ct = TestContext.Current.CancellationToken;
        try
        {
            string path = Path.Combine(directory, "encoder.onemodel");
            new NemotronAsrGraphBuilder(source, dimensions).Encoder(path, 0, 2, true, ct);
            using var model = Session.Load(path, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            float[] mel = Enumerable.Range(0, 16 * 57).Select(i => MathF.Sin(i * .1f)).ToArray();
            float[] mask = new float[8 * 8];
            for (int q = 0; q < 8; q++) for (int k = 0; k < 8; k++) if (k / 4 > q / 4) mask[q * 8 + k] = float.NegativeInfinity;
            float[] Run()
            {
                ct.ThrowIfCancellationRequested();
                using var input = Tensor.FromSpan<float>(mel, Shape.Create(1, 16, 57));
                using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, 8, 8));
                using var output = model.Run(new Dictionary<string, Tensor> { ["features"] = input, ["attention_mask"] = maskInput });
                return output["encoded"].Data<float>().ToArray();
            }
            var first = Run();
            for (int channel = 0; channel < 16; channel++) for (int frame = 32; frame < 57; frame++) mel[channel * 57 + frame] += 10;
            var changed = Run();
            Assert.Equal(64, first.Length);
            for (int channel = 0; channel < 8; channel++) for (int frame = 0; frame < 4; frame++) Assert.Equal(first[channel * 8 + frame], changed[channel * 8 + frame]);
            Assert.Contains(Enumerable.Range(0, first.Length), i => first[i] != changed[i]);
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(false, 4)]
    [InlineData(false, 20)]
    [InlineData(true, 20)]
    public void ExplicitCachesMatchWholeSequenceChunkAttention(bool subsamplingOnly, int chunks)
    {
        var dimensions = new FastConformerDimensions(MelBins: 16, Width: 8, Layers: 2, Heads: 2, Expansion: 2, ConvolutionKernel: 3, SubsamplingChannels: 32);
        var source = new FastConformerGraphTests.SyntheticTensors(dimensions, causal: true);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-caches-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var ct = TestContext.Current.CancellationToken;
        SessionOutputs? previous = null;
        List<Tensor> initial = [];
        try
        {
            string offline = Path.Combine(directory, "offline.onemodel"), streaming = Path.Combine(directory, "step.onemodel");
            if (subsamplingOnly)
            {
                new NemotronAsrGraphBuilder(source, dimensions).Subsampling(offline, false, ct);
                new NemotronAsrGraphBuilder(source, dimensions).Subsampling(streaming, true, ct);
            }
            else
            {
                new NemotronAsrGraphBuilder(source, dimensions).Encoder(offline, 0, 2, true, ct);
                new NemotronAsrGraphBuilder(source, dimensions).EncoderStep(streaming, 0, 2, true, ct);
            }
            using var complete = Session.Load(offline, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var step = Session.Load(streaming, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            const int chunkFrames = 4;
            int encodedFrames = chunks * chunkFrames, frames = encodedFrames * 8 - 7;
            float[] mel = Enumerable.Range(0, 16 * frames).Select(i => MathF.Sin(i * .07f)).ToArray();
            float[] mask = new float[encodedFrames * encodedFrames];
            for (int q = 0; q < encodedFrames; q++) for (int k = 0; k < encodedFrames; k++) if (k / chunkFrames > q / chunkFrames || k < q / chunkFrames * chunkFrames - 56) mask[q * encodedFrames + k] = float.NegativeInfinity;
            using var audio = Tensor.FromSpan<float>(mel, Shape.Create(1, 16, frames));
            using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, encodedFrames, encodedFrames));
            var fullInputs = new Dictionary<string, Tensor> { ["features"] = audio };
            if (!subsamplingOnly) fullInputs["attention_mask"] = maskInput;
            using var full = complete.Run(fullInputs);
            float[] expected = full["encoded"].Data<float>().ToArray(), actual = new float[expected.Length];
            var inputs = new Dictionary<string, Tensor>();
            void Zero(string name, params int[] shape)
            {
                var tensor = Tensor.Create(DType.F32, Shape.Create(shape), zero: true); initial.Add(tensor); inputs.Add(name, tensor);
            }
            int frequency = 16;
            for (int stage = 0; stage < 3; stage++) { Zero("subsampling_cache" + stage, 1, stage == 0 ? 1 : 32, 2, frequency); frequency = frequency / 2 + 1; }
            for (int layer = 0; layer < (subsamplingOnly ? 0 : 2); layer++)
            {
                Zero("key_cache" + layer, 1, 2, 56, 4); Zero("value_cache" + layer, 1, 2, 56, 4); Zero("convolution_cache" + layer, 1, 8, 2);
            }
            int offset = 0;
            for (int chunk = 0; chunk < chunks; chunk++)
            {
                ct.ThrowIfCancellationRequested();
                int count = chunk == 0 ? 25 : 32;
                float[] block = new float[16 * count], stepMask = new float[chunkFrames * (56 + chunkFrames)];
                for (int channel = 0; channel < 16; channel++) mel.AsSpan(channel * frames + offset, count).CopyTo(block.AsSpan(channel * count));
                for (int q = 0; q < chunkFrames; q++) for (int k = 0; k < 56 - chunk * chunkFrames; k++) stepMask[q * (56 + chunkFrames) + k] = float.NegativeInfinity;
                using var blockInput = Tensor.FromSpan<float>(block, Shape.Create(1, 16, count));
                using var stepMaskInput = Tensor.FromSpan<float>(stepMask, Shape.Create(1, 1, chunkFrames, 56 + chunkFrames));
                inputs["features"] = blockInput; if (!subsamplingOnly) inputs["attention_mask"] = stepMaskInput;
                if (previous != null)
                {
                    for (int stage = 0; stage < 3; stage++) inputs["subsampling_cache" + stage] = previous["next_subsampling_cache" + stage];
                    for (int layer = 0; layer < (subsamplingOnly ? 0 : 2); layer++) foreach (string kind in new[] { "key_cache", "value_cache", "convolution_cache" }) inputs[kind + layer] = previous["next_" + kind + layer];
                }
                var current = step.Run(inputs);
                var values = current["encoded"].Data<float>();
                Assert.Equal(8 * chunkFrames, values.Length);
                for (int channel = 0; channel < 8; channel++) values.Slice(channel * chunkFrames, chunkFrames).CopyTo(actual.AsSpan(channel * encodedFrames + chunk * chunkFrames));
                previous?.Dispose(); previous = current; offset += count;
            }
            Assert.Equal(frames, offset);
            for (int i = 0; i < actual.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5, $"index {i}, expected {expected[i]}, actual {actual[i]}");
        }
        finally { previous?.Dispose(); foreach (var tensor in initial) tensor.Dispose(); Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(1, 64)]
    [InlineData(2, 19)]
    [InlineData(4, 1)]
    [InlineData(4, 2)]
    [InlineData(4, 7)]
    [InlineData(4, 8)]
    [InlineData(4, 24)]
    [InlineData(4, 25)]
    [InlineData(4, 26)]
    [InlineData(4, 121)]
    [InlineData(4, 124)]
    [InlineData(4, 640)]
    [InlineData(7, 137)]
    [InlineData(14, 109)]
    public void ReusableCallerOwnedCachesMatchOfflineAndFlushPartialTail(int chunk, int frames)
    {
        var dimensions = new FastConformerDimensions(MelBins: 16, Width: 8, Layers: 2, Heads: 2, Expansion: 2, ConvolutionKernel: 3, SubsamplingChannels: 32);
        var source = new FastConformerGraphTests.SyntheticTensors(dimensions, causal: true);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-owned-cache-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory); var ct = TestContext.Current.CancellationToken;
        try
        {
            string offline = Path.Combine(directory, "all.onemodel"), first = Path.Combine(directory, "first.onemodel"), last = Path.Combine(directory, "last.onemodel"), tail = Path.Combine(directory, "tail.onemodel");
            new NemotronAsrGraphBuilder(source, dimensions).Encoder(offline, 0, 2, true, ct);
            new NemotronAsrGraphBuilder(source, dimensions).EncoderStep(first, 0, 1, true, ct);
            new NemotronAsrGraphBuilder(source, dimensions).EncoderStep(last, 1, 1, false, ct);
            new NemotronAsrGraphBuilder(source, dimensions).EncoderStep(tail, 0, 1, true, ct, finalSubsampling: true);
            using var all = Session.Load(offline, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var a = Session.Load(first, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var b = Session.Load(last, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var tailSession = Session.Load(tail, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            int encodedFrames = ((frames / 2 + 1) / 2 + 1) / 2 + 1;
            float[] mel = Enumerable.Range(0, 16 * frames).Select(i => MathF.Sin(i * .07f)).ToArray(), mask = new float[encodedFrames * encodedFrames];
            for (int q = 0; q < encodedFrames; q++) for (int k = 0; k < encodedFrames; k++) if (k < q / chunk * chunk - 56 || k >= q / chunk * chunk + chunk) mask[q * encodedFrames + k] = float.NegativeInfinity;
            using var audio = Tensor.FromSpan<float>(mel, Shape.Create(1, 16, frames));
            using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, encodedFrames, encodedFrames));
            using var result = all.Run(new Dictionary<string, Tensor> { ["features"] = audio, ["attention_mask"] = maskInput });
            float[] expected = result["encoded"].Data<float>().ToArray(), actual = new float[expected.Length]; int used = 0;
            void Emit(ReadOnlySpan<float> values, int valid, CancellationToken cancellation)
            {
                cancellation.ThrowIfCancellationRequested();
                for (int channel = 0; channel < 8; channel++) values.Slice(channel * chunk, valid).CopyTo(actual.AsSpan(channel * encodedFrames + used));
                used += valid;
            }
            using var state = new OneSpeech.FastConformer.NemotronAsrEncoderState(a, b, tailSession, dimensions, chunk, Emit, ct);
            float[] frame = new float[16];
            for (int pass = 0; pass < 2; pass++)
            {
                if (pass != 0) { state.Reset(); used = 0; Array.Clear(actual); }
                for (int t = 0; t < frames; t++) { for (int channel = 0; channel < 16; channel++) frame[channel] = mel[channel * frames + t]; state.Append(frame, ct); }
                state.Complete(ct); Assert.Equal(encodedFrames, used); Assert.Equal(encodedFrames, state.EncodedFrames);
                for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5, $"index {i}: expected {expected[i]}, actual {actual[i]}");
            }
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("encoder.subsampling.0.weight", "encoder.subsampling.conv_in.weight")]
    [InlineData("encoder.subsampling.2.pointwise.bias", "encoder.subsampling.layers.1.pointwise_conv.bias")]
    [InlineData("encoder.layers.10.conv.layer_norm.weight", "encoder.layers.10.conv.norm.weight")]
    [InlineData("encoder.layers.1.attention.u", "encoder.layers.1.self_attn.bias_u")]
    public void PublisherNamesAreMappedWithoutExecutableModelCode(string canonical, string publisher) => Assert.Equal(publisher, NemotronAsrTensorSource.PublisherName(canonical));
}
