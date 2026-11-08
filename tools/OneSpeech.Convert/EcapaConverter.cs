using static SpeakerGraph;

// Desplanques et al. 2020 §§3.1–3.3; geometry comes from the pinned hyperparams and tensor shapes.
internal sealed class EcapaConverter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private readonly HashSet<string> added = new(StringComparer.Ordinal);
    private string W(string name) { if (added.Add(name)) { var t = tensors[name]; g.Float(name, t.Shape, t.Values); } return name; }
    private string Conv(string x, string p, int dilation = 1)
    {
        int pad = tensors[p + ".weight"].Shape[^1] / 2 * dilation;
        if (pad > 0) x = g.Op("Pad", [x, g.Long(0, 0, pad, 0, 0, pad)], S("mode", "reflect"));
        string[] inputs = tensors.ContainsKey(p + ".bias") ? [x, W(p + ".weight"), W(p + ".bias")] : [x, W(p + ".weight")];
        return g.Op("Conv", inputs, Is("dilations", dilation), Is("pads", 0, 0));
    }
    private string Bn(string x, string p) => g.Op("BatchNormalization", [x, W(p + ".weight"), W(p + ".bias"), W(p + ".running_mean"), W(p + ".running_var")], F("epsilon", 1e-5f));
    private string Relu(string x) => g.Op("Relu", [x]);
    private string Tdnn(string x, string p, int dilation = 1)
    {
        x = Conv(x, p + ".conv.conv", dilation);
        return Bn(Relu(x), p + ".norm.norm");
    }
    private string Slice(string x, int start, int end) => g.Op("Slice", [x, g.Long(start), g.Long(end), g.Long(1)]);
    private string Sum(string x) => g.Op("ReduceSum", [x, g.Long(2)], I("keepdims", 1));
    private string Floor(string x, string name) => g.Op("Max", [x, g.Float(name, [1], [1e-12f])]);
    private string Expand(string x, string target) => g.Op("Expand", [x, g.Op("Shape", [target])]);
    internal void Save(string path)
    {
        string x = Tdnn("features", "blocks.0");
        var outputs = new List<string>();
        for (int block = 1; block <= 3; block++)
        {
            string p = $"blocks.{block}";
            string skip = x;
            string h = Tdnn(x, p + ".tdnn1");
            var chunks = new List<string>();
            string previous = "";
            for (int split = 0; split < 8; split++)
            {
                string part = Slice(h, split * 128, (split + 1) * 128);
                bool bypass = split == 0;
                if (!bypass)
                {
                    if (previous.Length > 0) part = g.Op("Add", [part, previous]);
                    int index = split - 1;
                    part = Tdnn(part, p + $".res2net_block.blocks.{index}", block + 1);
                    previous = part;
                }
                chunks.Add(part);
            }
            h = Tdnn(g.Concat(chunks.ToArray()), p + ".tdnn2");
            string scale = g.Op("Sigmoid", [Conv(Relu(Conv(g.Mean(h, 2), p + ".se_block.conv1.conv")), p + ".se_block.conv2.conv")]);
            x = g.Op("Add", [skip, g.Op("Mul", [h, scale])]);
            outputs.Add(x);
        }
        x = Tdnn(g.Concat(outputs.ToArray()), "mfa");
        string mean = g.Mean(x, 2), centered = g.Op("Sub", [x, mean]);
        string std = g.Op("Sqrt", [Floor(g.Mean(g.Op("Mul", [centered, centered]), 2), "global_epsilon")]);
        string context = g.Concat(x, Expand(mean, x), Expand(std, x));
        string attention = g.Op("Softmax", [Conv(g.Op("Tanh", [Tdnn(context, "asp.tdnn")]), "asp.conv.conv")], I("axis", 2));
        string weightedMean = Sum(g.Op("Mul", [x, attention]));
        centered = g.Op("Sub", [x, weightedMean]);
        string variance = Sum(g.Op("Mul", [g.Op("Mul", [centered, centered]), attention]));
        string pooled = g.Concat(weightedMean, g.Op("Sqrt", [Floor(variance, "weighted_epsilon")]));
        x = Conv(Bn(pooled, "asp_bn.norm"), "fc.conv");
        g.Save(path, g.Reshape(x, 1, 192), 192);
    }
}
