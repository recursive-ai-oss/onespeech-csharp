using static SpeakerGraph;

// Wang et al. 2023 (WeSpeaker) ResNet34 speaker encoder with the published pooling split: this graph ends at the
// frame-level feature map; statistics pooling and the final linear layer run in WeSpeakerEmbedder because the
// community pipeline weights the pooled frames by the speaker's activity.
// Layout read from the published checkpoint tensor names and shapes: a 3x3 stem with 32 channels, four stages of
// 3, 4, 6 and 3 basic blocks with 32, 64, 128 and 256 channels, stride 2 in time and frequency at the first block of
// stages 2-4, giving 256 x 10 frequency rows after an 80-band input.
internal sealed class WeSpeakerResNetConverter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private readonly HashSet<string> added = new(StringComparer.Ordinal);

    internal static readonly int[] BlocksPerStage = [3, 4, 6, 3];

    private string Weight(string name)
    {
        if (added.Add(name))
        {
            if (!tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException("The embedding checkpoint lacks tensor " + name + ".");
            g.Float(name, tensor.Shape, tensor.Values);
        }
        return name;
    }

    private string Conv(string x, string prefix, int stride)
    {
        var shape = tensors.TryGetValue(prefix + ".weight", out var t) ? t.Shape : throw new InvalidDataException("The embedding checkpoint lacks tensor " + prefix + ".weight.");
        if (shape.Length != 4 || shape[2] != shape[3]) throw new InvalidDataException("Unexpected convolution shape.");
        int pad = shape[2] / 2;
        return g.Op("Conv", [x, Weight(prefix + ".weight")], Is("strides", stride, stride), Is("pads", pad, pad, pad, pad));
    }

    private string Bn(string x, string prefix) =>
        g.Op("BatchNormalization", [x, Weight(prefix + ".weight"), Weight(prefix + ".bias"), Weight(prefix + ".running_mean"), Weight(prefix + ".running_var")], F("epsilon", 1e-5f));

    private string Relu(string x) => g.Op("Relu", [x]);

    private string Block(string x, string prefix, int stride)
    {
        string y = Relu(Bn(Conv(x, prefix + ".conv1", stride), prefix + ".bn1"));
        y = Bn(Conv(y, prefix + ".conv2", 1), prefix + ".bn2");
        string skip = tensors.ContainsKey(prefix + ".shortcut.0.weight") ? Bn(Conv(x, prefix + ".shortcut.0", stride), prefix + ".shortcut.1") : x;
        return Relu(g.Op("Add", [y, skip]));
    }

    /// <summary>Writes the frame-level graph: features [1, 80, T] to frames [1, 2560, ceil(T / 8)].</summary>
    internal void Save(string path)
    {
        string x = g.Op("Unsqueeze", ["features", g.Long(1)]);
        x = Relu(Bn(Conv(x, "resnet.conv1", 1), "resnet.bn1"));
        for (int stage = 0; stage < BlocksPerStage.Length; stage++)
            for (int block = 0; block < BlocksPerStage[stage]; block++)
                x = Block(x, $"resnet.layer{stage + 1}.{block}", stage > 0 && block == 0 ? 2 : 1);
        x = g.Reshape(x, 1, 2560, -1);
        g.Save(path, x, 2560, outputName: "frames", outputShape: [1, 2560, "steps"]);
    }

    /// <summary>Returns the final linear layer (weights [256, 5120] row-major, bias [256]).</summary>
    internal (float[] Weights, float[] Bias) Projection()
    {
        var w = tensors["resnet.seg_1.weight"]; var b = tensors["resnet.seg_1.bias"];
        if (w.Shape.Length != 2 || w.Shape[1] != 5120 || b.Shape.Length != 1 || b.Shape[0] != w.Shape[0]) throw new InvalidDataException("Unexpected embedding projection shape.");
        return (w.Values, b.Values);
    }
}
