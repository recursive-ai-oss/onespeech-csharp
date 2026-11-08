using OneSpeech.Formats;
using static OneSpeech.Catalog.Conversion.FastConformerGraph;

namespace OneSpeech.Catalog.Conversion;

// Cache-aware FastConformer: Noroozi et al. 2024 §3, with the publisher's causal and chunk-aware configuration.
internal sealed class NemotronAsrGraphBuilder
{
    private readonly FastConformerDimensions dimensions;
    private readonly FastConformerGraph graph;

    internal NemotronAsrGraphBuilder(IFloatTensorSource tensors, FastConformerDimensions dimensions)
    {
        ArgumentNullException.ThrowIfNull(tensors); ArgumentNullException.ThrowIfNull(dimensions);
        dimensions.Validate(); this.dimensions = dimensions; graph = new(tensors);
    }

    // FastConformer §2: 8x depthwise-separable subsampling followed by Conformer blocks.
    // Splitting at layer boundaries keeps each serialized graph below the runtime's 2 GiB limit.
    internal void Encoder(string path, int firstLayer, int layerCount, bool subsample, CancellationToken cancellationToken, bool probeOutputs = false)
    {
        if (firstLayer < 0 || layerCount < 1 || firstLayer > dimensions.Layers - layerCount || subsample && firstLayer != 0) throw new ArgumentException("Invalid encoder layer range.");
        int d = dimensions.Width;
        graph.Input("features", 1, subsample ? dimensions.MelBins : d, "frames");
        graph.Input("attention_mask", 1, 1, "encoded_frames", "encoded_frames");
        string x = subsample ? Subsample("features") : graph.Transpose("features", 0, 2, 1);
        if (probeOutputs) graph.Output(x, "hidden" + firstLayer, 1, "encoded_frames", d);
        var (positions, indices) = RelativePositions(x);
        for (int layer = firstLayer; layer < firstLayer + layerCount; layer++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string prefix = "encoder.layers." + layer;
            // Gulati et al. 2020 Eq.1: two half-step feed-forwards sandwich attention and convolution.
            x = FeedForward(x, prefix + ".ff1");
            x = graph.Op("Add", [x, Attention(graph.Norm(x, prefix + ".attention_norm", d), prefix + ".attention", positions, indices)]);
            x = graph.Op("Add", [x, Convolution(graph.Norm(x, prefix + ".conv_norm", d), prefix + ".conv")]);
            x = graph.Norm(FeedForward(x, prefix + ".ff2"), prefix + ".final_norm", d);
            if (probeOutputs) graph.Output(x, "hidden" + (layer + 1), 1, "encoded_frames", d);
        }
        graph.Output(graph.Transpose(x, 0, 2, 1), "encoded", 1, d, "encoded_frames");
        graph.Save(path, cancellationToken);
    }

    internal void PromptProjector(string path, CancellationToken cancellationToken)
    {
        graph.Input("prompt_input", 1, dimensions.Width + 128, "frames");
        string x = graph.Transpose("prompt_input", 0, 2, 1);
        x = graph.Op("Relu", [graph.Linear(x, "prompt_projector.linear_1", dimensions.Width + 128, 2048, bias: true)]);
        x = graph.Linear(x, "prompt_projector.linear_2", 2048, dimensions.Width, bias: true);
        graph.Output(graph.Transpose(x, 0, 2, 1), "conditioned", 1, dimensions.Width, "frames");
        graph.Save(path, cancellationToken);
    }

    internal void Subsampling(string path, bool cached, CancellationToken cancellationToken)
    {
        graph.Input("features", 1, dimensions.MelBins, "frames");
        string x = cached ? SubsampleStep("features") : Subsample("features");
        graph.Output(graph.Transpose(x, 0, 2, 1), "encoded", 1, dimensions.Width, "encoded_frames");
        graph.Save(path, cancellationToken);
    }

    // Noroozi et al. §3.3: ordinary tensor inputs/outputs carry attention and convolution state.
    // Projected keys/values are cached instead of re-projecting the same stored activations.
    internal void EncoderStep(string path, int firstLayer, int layerCount, bool subsample, CancellationToken cancellationToken, bool finalSubsampling = false)
    {
        if (firstLayer < 0 || layerCount < 1 || firstLayer > dimensions.Layers - layerCount || subsample && firstLayer != 0) throw new ArgumentException("Invalid encoder layer range.");
        int d = dimensions.Width;
        graph.Input("features", 1, subsample ? dimensions.MelBins : d, "frames");
        graph.Input("attention_mask", 1, 1, "query_frames", "key_frames");
        string x = subsample ? SubsampleStep("features", finalSubsampling) : graph.Transpose("features", 0, 2, 1);
        var (positions, indices) = RelativePositions(x, 56);
        for (int layer = firstLayer; layer < firstLayer + layerCount; layer++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string prefix = "encoder.layers." + layer;
            x = FeedForward(x, prefix + ".ff1");
            x = graph.Op("Add", [x, Attention(graph.Norm(x, prefix + ".attention_norm", d), prefix + ".attention", positions, indices, layer)]);
            x = graph.Op("Add", [x, Convolution(graph.Norm(x, prefix + ".conv_norm", d), prefix + ".conv", layer)]);
            x = graph.Norm(FeedForward(x, prefix + ".ff2"), prefix + ".final_norm", d);
        }
        graph.Output(graph.Transpose(x, 0, 2, 1), "encoded", 1, d, "encoded_frames");
        graph.Save(path, cancellationToken);
    }

    private string SubsampleStep(string x, bool finalPadding = false)
    {
        int c = dimensions.SubsamplingChannels, frequency = dimensions.MelBins;
        x = graph.Op("Unsqueeze", [graph.Transpose(x, 0, 2, 1), graph.Ints(1)]);
        for (int stage = 0; stage < 3; stage++)
        {
            string prefix = "encoder.subsampling." + stage, cache = "subsampling_cache" + stage;
            graph.Input(cache, 1, stage == 0 ? 1 : c, "subsampling_phase", frequency);
            x = graph.Op("Concat", [cache, x], A("axis", 2));
            // Causal stride-two convolutions start with two zero history frames. After the
            // first odd-length chunk, exactly one history frame preserves the stride phase.
            graph.Output(graph.Slice(x, 2, -1, int.MaxValue), "next_" + cache, 1, stage == 0 ? 1 : c, 1, frequency);
            if (stage == 0) x = graph.Op("Conv", [x, graph.Weight(prefix + ".weight", c, 1, 3, 3), graph.Weight(prefix + ".bias", c)], As("strides", 2, 2), As("pads", 0, 2, finalPadding ? 1 : 0, 1));
            else
            {
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", c, 1, 3, 3), graph.Weight(prefix + ".depthwise.bias", c)], A("group", c), As("strides", 2, 2), As("pads", 0, 2, finalPadding ? 1 : 0, 1));
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".pointwise.weight", c, c, 1, 1), graph.Weight(prefix + ".pointwise.bias", c)]);
            }
            x = graph.Op("Relu", [x]); frequency = frequency / 2 + 1;
        }
        x = graph.Reshape(graph.Transpose(x, 0, 2, 1, 3), 1, -1, c * frequency);
        return graph.Linear(x, "encoder.subsampling.projection", c * frequency, dimensions.Width, bias: true);
    }

    private string Subsample(string x)
    {
        int c = dimensions.SubsamplingChannels;
        x = graph.Op("Unsqueeze", [graph.Transpose(x, 0, 2, 1), graph.Ints(1)]);
        // End padding is applied at every strided layer, not just to the original mel
        // sequence. Observed native lengths follow f(f(f(L))), f(L)=floor(L/2)+1.
        for (int stage = 0; stage < 3; stage++)
        {
            string prefix = "encoder.subsampling." + stage;
            if (stage == 0) x = graph.Op("Conv", [x, graph.Weight(prefix + ".weight", c, 1, 3, 3), graph.Weight(prefix + ".bias", c)], As("strides", 2, 2), As("pads", 2, 2, 1, 1));
            else
            {
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", c, 1, 3, 3), graph.Weight(prefix + ".depthwise.bias", c)], A("group", c), As("strides", 2, 2), As("pads", 2, 2, 1, 1));
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".pointwise.weight", c, c, 1, 1), graph.Weight(prefix + ".pointwise.bias", c)]);
            }
            x = graph.Op("Relu", [x]);
        }
        int flattened = c * (dimensions.MelBins / 8 + 1);
        x = graph.Reshape(graph.Transpose(x, 0, 2, 1, 3), 1, -1, flattened);
        return graph.Linear(x, "encoder.subsampling.projection", flattened, dimensions.Width, bias: true);
    }

    private string FeedForward(string x, string prefix)
    {
        int d = dimensions.Width, expanded = d * dimensions.Expansion;
        string y = graph.Norm(x, prefix + ".norm", d);
        y = graph.Swish(graph.Linear(y, prefix + ".in", d, expanded));
        y = graph.Linear(y, prefix + ".out", expanded, d);
        return graph.Op("Add", [x, graph.Op("Mul", [y, graph.Scalar(.5f)])]);
    }

    private (string Positions, string Indices) RelativePositions(string x, int pastFrames = 0)
    {
        int d = dimensions.Width;
        string time = graph.Op("Gather", [graph.Op("Shape", [x]), graph.LongScalar(1)], A("axis", 0));
        string keyTime = pastFrames == 0 ? time : graph.Op("Add", [time, graph.LongScalar(pastFrames)]);
        string center = graph.Op("Sub", [keyTime, graph.LongScalar(1)]);
        string sequence = graph.Op("Range", [center, graph.Op("Neg", [time]), graph.LongScalar(-1)]);
        sequence = graph.Op("Cast", [sequence], A("to", 1));
        float[] inverseFrequency = new float[d / 2];
        for (int i = 0; i < inverseFrequency.Length; i++) inverseFrequency[i] = (float)Math.Exp(-Math.Log(10000) * 2 * i / d);
        string phase = graph.Op("Mul", [graph.Op("Unsqueeze", [sequence, graph.Ints(1)]), graph.Float(inverseFrequency, d / 2)]);
        string sin = graph.Op("Unsqueeze", [graph.Op("Sin", [phase]), graph.Ints(2)]);
        string cos = graph.Op("Unsqueeze", [graph.Op("Cos", [phase]), graph.Ints(2)]);
        string positions = graph.Reshape(graph.Op("Concat", [sin, cos], A("axis", 2)), 1, -1, d);
        string range = graph.Op("Range", [graph.LongScalar(0), time, graph.LongScalar(1)]);
        string keyRange = graph.Op("Range", [graph.LongScalar(0), keyTime, graph.LongScalar(1)]);
        string rows = graph.Op("Unsqueeze", [range, graph.Ints(1)]), columns = graph.Op("Unsqueeze", [keyRange, graph.Ints(0)]);
        // Cached keys precede the current query chunk. In the descending relative table,
        // query i and key j map to query_count-1-i+j, including the cached prefix.
        string queryCenter = graph.Op("Sub", [time, graph.LongScalar(1)]);
        string indices = graph.Op("Add", [graph.Op("Sub", [queryCenter, rows]), columns]);
        indices = graph.Op("Unsqueeze", [indices, graph.Ints(0, 1)]);
        string timeVector = graph.Op("Unsqueeze", [time, graph.Ints(0)]);
        string keyTimeVector = graph.Op("Unsqueeze", [keyTime, graph.Ints(0)]);
        string shape = graph.Op("Concat", [graph.Ints(1, dimensions.Heads), timeVector, keyTimeVector], A("axis", 0));
        return (positions, graph.Op("Expand", [indices, shape]));
    }

    private string Heads(string x) => graph.Transpose(graph.Reshape(x, 1, -1, dimensions.Heads, dimensions.Width / dimensions.Heads), 0, 2, 1, 3);

    private string Attention(string x, string prefix, string positions, string indices, int cacheLayer = -1)
    {
        int d = dimensions.Width, heads = dimensions.Heads, depth = d / heads;
        string q = Heads(graph.Linear(x, prefix + ".q", d, d)), k = Heads(graph.Linear(x, prefix + ".k", d, d)), v = Heads(graph.Linear(x, prefix + ".v", d, d));
        if (cacheLayer >= 0)
        {
            string keyCache = "key_cache" + cacheLayer, valueCache = "value_cache" + cacheLayer;
            graph.Input(keyCache, 1, heads, 56, depth); graph.Input(valueCache, 1, heads, 56, depth);
            k = graph.Op("Concat", [keyCache, k], A("axis", 2)); v = graph.Op("Concat", [valueCache, v], A("axis", 2));
            graph.Output(graph.Slice(k, 2, -56, int.MaxValue), "next_" + keyCache, 1, heads, 56, depth);
            graph.Output(graph.Slice(v, 2, -56, int.MaxValue), "next_" + valueCache, 1, heads, 56, depth);
        }
        string p = Heads(graph.Linear(positions, prefix + ".position", d, d));
        string u = graph.Reshape(graph.Weight(prefix + ".u", heads, depth), 1, heads, 1, depth);
        string b = graph.Reshape(graph.Weight(prefix + ".v_bias", heads, depth), 1, heads, 1, depth);
        // Transformer-XL §3.3: content and relative-position scores, with separate learned query biases.
        string content = graph.Op("MatMul", [graph.Op("Add", [q, u]), graph.Transpose(k, 0, 1, 3, 2)]);
        string relative = graph.Op("MatMul", [graph.Op("Add", [q, b]), graph.Transpose(p, 0, 1, 3, 2)]);
        relative = graph.Op("GatherElements", [relative, indices], A("axis", 3));
        string scores = graph.Op("Div", [graph.Op("Add", [content, relative]), graph.Scalar(MathF.Sqrt(depth))]);
        scores = graph.Op("Add", [scores, "attention_mask"]);
        string attended = graph.Op("MatMul", [graph.Op("Softmax", [scores], A("axis", -1)), v]);
        attended = graph.Reshape(graph.Transpose(attended, 0, 2, 1, 3), 1, -1, d);
        return graph.Linear(attended, prefix + ".out", d, d);
    }

    private string Convolution(string x, string prefix, int cacheLayer = -1)
    {
        int d = dimensions.Width, kernel = dimensions.ConvolutionKernel;
        x = graph.Transpose(x, 0, 2, 1);
        x = graph.Op("Conv", [x, graph.Weight(prefix + ".pointwise_in.weight", 2 * d, d, 1)]);
        x = graph.Op("Mul", [graph.Slice(x, 1, 0, d), graph.Op("Sigmoid", [graph.Slice(x, 1, d, 2 * d)])]);
        if (cacheLayer >= 0)
        {
            string cache = "convolution_cache" + cacheLayer;
            graph.Input(cache, 1, d, kernel - 1);
            x = graph.Op("Concat", [cache, x], A("axis", 2));
            graph.Output(graph.Slice(x, 2, -(kernel - 1), int.MaxValue), "next_" + cache, 1, d, kernel - 1);
            x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", d, 1, kernel)], A("group", d));
        }
        else
        x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", d, 1, kernel)], A("group", d), As("pads", kernel - 1, 0));
        x = graph.Transpose(graph.Norm(graph.Transpose(x, 0, 2, 1), prefix + ".layer_norm", d), 0, 2, 1);
        x = graph.Op("Conv", [graph.Swish(x), graph.Weight(prefix + ".pointwise_out.weight", d, d, 1)]);
        return graph.Transpose(x, 0, 2, 1);
    }

}
