using OneSpeech.Formats;
using static OneSpeech.Catalog.Conversion.FastConformerGraph;

namespace OneSpeech.Catalog.Conversion;

// Canonical tensor names here are a converter schema, not an assumption about publisher container names.
// A tensor-source adapter maps publisher names to these names before graph construction.
internal sealed record FastConformerDimensions(int MelBins = 128, int Width = 1024, int Layers = 24, int Heads = 8,
    int Expansion = 4, int ConvolutionKernel = 9, int SubsamplingChannels = 256, int PredictorWidth = 640,
    int PredictorLayers = 2, int JointWidth = 640, int Vocabulary = 8193, int DurationCount = 5)
{
    internal void Validate()
    {
        if (MelBins is < 8 or > 256 || Width is < 2 or > 2048 || Width % 2 != 0 || Layers is < 1 or > 64 || Heads < 1 || Width % Heads != 0 ||
            Expansion is < 1 or > 8 || ConvolutionKernel is < 1 or > 127 || ConvolutionKernel % 2 != 1 || SubsamplingChannels is < 1 or > 1024 ||
            PredictorWidth is < 1 or > 2048 || PredictorLayers is < 1 or > 8 || JointWidth is < 1 or > 2048 || Vocabulary is < 2 or > 65536 || DurationCount is < 0 or > 1024)
            throw new ArgumentException("Invalid FastConformer dimensions.");
    }
}

internal sealed class FastConformerGraphBuilder
{
    private readonly FastConformerDimensions dimensions;
    private readonly FastConformerGraph graph;

    internal FastConformerGraphBuilder(IFloatTensorSource tensors, FastConformerDimensions dimensions)
    {
        ArgumentNullException.ThrowIfNull(tensors); ArgumentNullException.ThrowIfNull(dimensions);
        dimensions.Validate(); this.dimensions = dimensions; graph = new(tensors);
    }

    // FastConformer §2: 8x depthwise-separable subsampling followed by Conformer blocks.
    // Splitting at layer boundaries keeps each serialized graph below the runtime's 2 GiB limit.
    internal void Encoder(string path, int firstLayer, int layerCount, bool subsample, CancellationToken cancellationToken)
    {
        if (firstLayer < 0 || layerCount < 1 || firstLayer > dimensions.Layers - layerCount || subsample && firstLayer != 0) throw new ArgumentException("Invalid encoder layer range.");
        int d = dimensions.Width;
        graph.Input("features", 1, subsample ? dimensions.MelBins : d, "frames");
        string x = subsample ? Subsample("features") : graph.Transpose("features", 0, 2, 1);
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
        }
        graph.Output(graph.Transpose(x, 0, 2, 1), "encoded", 1, d, "encoded_frames");
        graph.Save(path, cancellationToken);
    }

    private string Subsample(string x)
    {
        int c = dimensions.SubsamplingChannels;
        x = graph.Op("Unsqueeze", [x, graph.Ints(1)]);
        for (int stage = 0; stage < 3; stage++)
        {
            string prefix = "encoder.subsampling." + stage;
            if (stage == 0) x = graph.Op("Conv", [x, graph.Weight(prefix + ".weight", c, 1, 3, 3), graph.Weight(prefix + ".bias", c)], As("strides", 2, 2), As("pads", 1, 1, 1, 1));
            else
            {
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", c, 1, 3, 3), graph.Weight(prefix + ".depthwise.bias", c)], A("group", c), As("strides", 2, 2), As("pads", 1, 1, 1, 1));
                x = graph.Op("Conv", [x, graph.Weight(prefix + ".pointwise.weight", c, c, 1, 1), graph.Weight(prefix + ".pointwise.bias", c)]);
            }
            x = graph.Op("Relu", [x]);
        }
        int flattened = c * ((dimensions.MelBins + 7) / 8);
        x = graph.Reshape(graph.Transpose(x, 0, 3, 1, 2), 1, -1, flattened);
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

    private (string Positions, string Indices) RelativePositions(string x)
    {
        int d = dimensions.Width;
        string time = graph.Op("Gather", [graph.Op("Shape", [x]), graph.LongScalar(1)], A("axis", 0));
        string center = graph.Op("Sub", [time, graph.LongScalar(1)]);
        string sequence = graph.Op("Range", [center, graph.Op("Neg", [time]), graph.LongScalar(-1)]);
        sequence = graph.Op("Cast", [sequence], A("to", 1));
        float[] inverseFrequency = new float[d / 2];
        for (int i = 0; i < inverseFrequency.Length; i++) inverseFrequency[i] = (float)Math.Exp(-Math.Log(10000) * 2 * i / d);
        string phase = graph.Op("Mul", [graph.Op("Unsqueeze", [sequence, graph.Ints(1)]), graph.Float(inverseFrequency, d / 2)]);
        string sin = graph.Op("Unsqueeze", [graph.Op("Sin", [phase]), graph.Ints(2)]);
        string cos = graph.Op("Unsqueeze", [graph.Op("Cos", [phase]), graph.Ints(2)]);
        string positions = graph.Reshape(graph.Op("Concat", [sin, cos], A("axis", 2)), 1, -1, d);
        string range = graph.Op("Range", [graph.LongScalar(0), time, graph.LongScalar(1)]);
        string rows = graph.Op("Unsqueeze", [range, graph.Ints(1)]), columns = graph.Op("Unsqueeze", [range, graph.Ints(0)]);
        // Relative position i-j is stored at center-i+j in a descending signed position table.
        string indices = graph.Op("Add", [graph.Op("Sub", [center, rows]), columns]);
        indices = graph.Op("Unsqueeze", [indices, graph.Ints(0, 1)]);
        string timeVector = graph.Op("Unsqueeze", [time, graph.Ints(0)]);
        string shape = graph.Op("Concat", [graph.Ints(1, dimensions.Heads), timeVector, timeVector], A("axis", 0));
        return (positions, graph.Op("Expand", [indices, shape]));
    }

    private string Heads(string x) => graph.Transpose(graph.Reshape(x, 1, -1, dimensions.Heads, dimensions.Width / dimensions.Heads), 0, 2, 1, 3);

    private string Attention(string x, string prefix, string positions, string indices)
    {
        int d = dimensions.Width, heads = dimensions.Heads, depth = d / heads;
        string q = Heads(graph.Linear(x, prefix + ".q", d, d)), k = Heads(graph.Linear(x, prefix + ".k", d, d)), v = Heads(graph.Linear(x, prefix + ".v", d, d));
        string p = Heads(graph.Linear(positions, prefix + ".position", d, d));
        string u = graph.Reshape(graph.Weight(prefix + ".u", heads, depth), 1, heads, 1, depth);
        string b = graph.Reshape(graph.Weight(prefix + ".v_bias", heads, depth), 1, heads, 1, depth);
        // Transformer-XL §3.3: content and relative-position scores, with separate learned query biases.
        string content = graph.Op("MatMul", [graph.Op("Add", [q, u]), graph.Transpose(k, 0, 1, 3, 2)]);
        string relative = graph.Op("MatMul", [graph.Op("Add", [q, b]), graph.Transpose(p, 0, 1, 3, 2)]);
        relative = graph.Op("GatherElements", [relative, indices], A("axis", 3));
        string scores = graph.Op("Div", [graph.Op("Add", [content, relative]), graph.Scalar(MathF.Sqrt(depth))]);
        string attended = graph.Op("MatMul", [graph.Op("Softmax", [scores], A("axis", -1)), v]);
        attended = graph.Reshape(graph.Transpose(attended, 0, 2, 1, 3), 1, -1, d);
        return graph.Linear(attended, prefix + ".out", d, d);
    }

    private string Convolution(string x, string prefix)
    {
        int d = dimensions.Width, kernel = dimensions.ConvolutionKernel;
        x = graph.Transpose(x, 0, 2, 1);
        x = graph.Op("Conv", [x, graph.Weight(prefix + ".pointwise_in.weight", 2 * d, d, 1)]);
        x = graph.Op("Mul", [graph.Slice(x, 1, 0, d), graph.Op("Sigmoid", [graph.Slice(x, 1, d, 2 * d)])]);
        x = graph.Op("Conv", [x, graph.Weight(prefix + ".depthwise.weight", d, 1, kernel)], A("group", d), As("pads", kernel / 2, kernel / 2));
        x = graph.Op("BatchNormalization", [x, graph.Weight(prefix + ".batch_norm.weight", d), graph.Weight(prefix + ".batch_norm.bias", d), graph.Weight(prefix + ".batch_norm.mean", d), graph.Weight(prefix + ".batch_norm.variance", d)], F("epsilon", 1e-5f));
        x = graph.Op("Conv", [graph.Swish(x), graph.Weight(prefix + ".pointwise_out.weight", d, d, 1)]);
        return graph.Transpose(x, 0, 2, 1);
    }

    // Packed input/output keeps predictor state explicit using ordinary graph inputs, with no hidden model state.
    // Layout is [embedding, h0, c0, h1, c1, ...]; output replaces embedding with the final predictor vector.
    internal void Predictor(string path, CancellationToken cancellationToken)
    {
        int d = dimensions.PredictorWidth;
        graph.Input("predictor_state", 1, d * (1 + 2 * dimensions.PredictorLayers));
        string x = graph.Slice("predictor_state", 1, 0, d);
        List<string> states = [];
        for (int layer = 0; layer < dimensions.PredictorLayers; layer++)
        {
            string prefix = "predictor.layers." + layer;
            string h = graph.Slice("predictor_state", 1, d * (1 + 2 * layer), d * (2 + 2 * layer));
            string c = graph.Slice("predictor_state", 1, d * (2 + 2 * layer), d * (3 + 2 * layer));
            string gates = graph.Op("Add", [graph.Linear(x, prefix + ".input", d, 4 * d, bias: true), graph.Linear(h, prefix + ".hidden", d, 4 * d, bias: true)]);
            string i = graph.Op("Sigmoid", [graph.Slice(gates, 1, 0, d)]), f = graph.Op("Sigmoid", [graph.Slice(gates, 1, d, 2 * d)]);
            string g = graph.Op("Tanh", [graph.Slice(gates, 1, 2 * d, 3 * d)]), o = graph.Op("Sigmoid", [graph.Slice(gates, 1, 3 * d, 4 * d)]);
            c = graph.Op("Add", [graph.Op("Mul", [f, c]), graph.Op("Mul", [i, g])]);
            x = graph.Op("Mul", [o, graph.Op("Tanh", [c])]);
            states.Add(x); states.Add(c);
        }
        graph.Output(graph.Op("Concat", [x, .. states], A("axis", 1)), "next_predictor_state", 1, d * (1 + 2 * dimensions.PredictorLayers));
        graph.Save(path, cancellationToken);
    }

    // The last projection contains token logits (including blank), followed by duration logits for TDT.
    internal void Joint(string path, CancellationToken cancellationToken)
    {
        int encoder = dimensions.Width, predictor = dimensions.PredictorWidth, hidden = dimensions.JointWidth;
        graph.Input("joint_input", 1, encoder + predictor);
        string f = graph.Linear(graph.Slice("joint_input", 1, 0, encoder), "joint.encoder", encoder, hidden, bias: true);
        string g = graph.Linear(graph.Slice("joint_input", 1, encoder, encoder + predictor), "joint.predictor", predictor, hidden, bias: true);
        string logits = graph.Linear(graph.Op("Relu", [graph.Op("Add", [f, g])]), "joint.output", hidden, dimensions.Vocabulary + dimensions.DurationCount, bias: true);
        graph.Output(logits, "logits", 1, dimensions.Vocabulary + dimensions.DurationCount);
        graph.Save(path, cancellationToken);
    }
}
