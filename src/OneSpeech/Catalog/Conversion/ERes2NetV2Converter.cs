using static SpeakerGraph;

// Chen et al. 2024 (ERes2NetV2, arXiv 2406.02167) §2.1-2.3: a residual stem, four stages of bottleneck-like
// blocks (1x1 reduce, two 3x3 splits joined by an attentional fusion in stages 3 and 4, 1x1 expand), a
// bottom-up fusion of stage 3 and stage 4, statistics pooling and one linear layer.
// Details the paper leaves open were settled by observing the released checkpoint's stage outputs: the stride
// sits in each block's first 1x1 convolution and its shortcut, the fusion is x*(1+a)+y*(1-a) with a = tanh(att),
// the in-block fusion is followed by ReLU and the stage 3/4 fusion is not, and the pooled mean and unbiased
// standard deviation are taken over time of the channel-by-frequency map.
internal sealed class ERes2NetV2Converter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private string? clipMin, clipMax;
    private readonly HashSet<string> added = new(StringComparer.Ordinal);
    private string Weight(string name)
    {
        if (added.Add(name)) { var tensor = tensors[name]; g.Float(name, tensor.Shape, tensor.Values); }
        return name;
    }
    private string Conv(string x, string prefix, int stride = 1)
    {
        string w = prefix + ".weight";
        var shape = tensors[w].Shape;
        string[] inputs = tensors.ContainsKey(prefix + ".bias") ? [x, Weight(w), Weight(prefix + ".bias")] : [x, Weight(w)];
        return g.Op("Conv", inputs, Is("strides", stride, stride), Is("pads", shape[2] / 2, shape[3] / 2, shape[2] / 2, shape[3] / 2));
    }
    private string Bn(string x, string prefix) =>
        g.Op("BatchNormalization", [x, Weight(prefix + ".weight"), Weight(prefix + ".bias"), Weight(prefix + ".running_mean"), Weight(prefix + ".running_var")], F("epsilon", 1e-5f));
    private string Relu(string x) => g.Op("Relu", [x]);
    private string Slice(string x, long from, long to) => g.Op("Slice", [x, g.Long(from), g.Long(to), g.Long(1)]);

    // Attentional feature fusion (paper eq. 1-2): Att = tanh(BN(W2 SiLU(BN(W1 [x, y])))).
    private string Fuse(string x, string y, string prefix, bool relu)
    {
        string h = Bn(Conv(g.Concat(x, y), prefix + ".local_att.0"), prefix + ".local_att.1");
        h = g.Op("Mul", [h, g.Op("Sigmoid", [h])]);
        string attention = g.Op("Tanh", [Bn(Conv(h, prefix + ".local_att.3"), prefix + ".local_att.4")]);
        string one = g.Float(prefix + ".one", [1], [1]);
        string fused = g.Op("Add", [g.Op("Mul", [x, g.Op("Add", [one, attention])]), g.Op("Mul", [y, g.Op("Sub", [one, attention])])]);
        return relu ? Relu(fused) : fused;
    }

    private string Block(string x, string prefix, int stride, bool fuse)
    {
        string reduced = Relu(Bn(Conv(x, prefix + ".conv1", stride), prefix + ".bn1"));
        int half = tensors[prefix + ".conv1.weight"].Shape[0] / 2;
        string first = Slice(reduced, 0, half), second = Slice(reduced, half, 2 * half);
        string y0 = Relu(Bn(Conv(first, prefix + ".convs.0"), prefix + ".bns.0"));
        string joined = fuse ? Fuse(y0, second, prefix + ".fuse_models.0", relu: true) : g.Op("Add", [y0, second]);
        string y1 = Relu(Bn(Conv(joined, prefix + ".convs.1"), prefix + ".bns.1"));
        string expanded = Bn(Conv(g.Concat(y0, y1), prefix + ".conv3"), prefix + ".bn3");
        string skip = tensors.ContainsKey(prefix + ".shortcut.0.weight") ? Bn(Conv(x, prefix + ".shortcut.0", stride), prefix + ".shortcut.1") : x;
        // Observed in the released checkpoint's block outputs: the activation after the residual sum is capped at 20.
        return g.Op("Clip", [g.Op("Add", [expanded, skip]), clipMin ??= g.Float("clip.min", [], [0]), clipMax ??= g.Float("clip.max", [], [20])]);
    }

    internal void Save(string path)
    {
        string x = g.Op("Unsqueeze", ["features", g.Long(1)]);
        x = Relu(Bn(Conv(x, "conv1"), "bn1"));
        int[] blocks = [3, 4, 6, 3];
        string stage3 = x;
        for (int layer = 1; layer <= 4; layer++)
        {
            for (int b = 0; b < blocks[layer - 1]; b++) x = Block(x, $"layer{layer}.{b}", layer > 1 && b == 0 ? 2 : 1, fuse: layer >= 3);
            if (layer == 3) stage3 = x;
        }
        // Bottom-up fusion of the down-sampled stage 3 and stage 4 (paper §2.2, eq. 1).
        x = Fuse(x, Conv(stage3, "layer3_ds", 2), "fuse34", relu: false);
        // Statistics pooling over time of the channel-by-frequency map: mean, then unbiased standard deviation.
        x = g.Reshape(x, 1, tensors["seg_1.weight"].Shape[1] / 2, -1);
        string mean = g.Mean(x, 2);
        string deviation = g.Op("Sub", [x, mean]);
        string variance = g.Mean(g.Op("Mul", [deviation, deviation]), 2);
        string length = g.Op("Cast", [g.Op("Gather", [g.Op("Shape", [x]), g.Long(2)], I("axis", 0))], I("to", 1));
        string unit = g.Float("pool.one", [1], [1]);
        variance = g.Op("Mul", [variance, g.Op("Div", [length, g.Op("Sub", [length, unit])])]);
        x = g.Reshape(g.Concat(mean, g.Op("Sqrt", [variance])), 1, -1);
        var weight = tensors["seg_1.weight"];
        int embedding = weight.Shape[0], inputs = weight.Shape[1];
        float[] transposed = new float[weight.Values.Length];
        for (int o = 0; o < embedding; o++) for (int i = 0; i < inputs; i++) transposed[i * embedding + o] = weight.Values[o * inputs + i];
        x = g.Op("Add", [g.Op("MatMul", [x, g.Float("seg_1.weight.t", [inputs, embedding], transposed)]), Weight("seg_1.bias")]);
        g.Save(path, x, embedding);
    }
}
