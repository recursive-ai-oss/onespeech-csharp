using static SpeakerGraph;

// Wang et al. 2023 §§2.1–2.4: FCM → dense TDNN/CAM → statistics pooling → embedding.
internal sealed class CampPlusConverter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private readonly HashSet<string> added = new(StringComparer.Ordinal);
    private string Weight(string name)
    {
        if (added.Add(name)) { var tensor = tensors[name]; g.Float(name, tensor.Shape, tensor.Values); }
        return name;
    }
    private string Conv(string x, string prefix, int stride = 1, int dilation = 1)
    {
        string w = prefix + ".weight";
        var shape = tensors[w].Shape;
        string[] inputs = tensors.ContainsKey(prefix + ".bias") ? [x, Weight(w), Weight(prefix + ".bias")] : [x, Weight(w)];
        int kernel = shape[^1];
        return shape.Length == 4
            ? g.Op("Conv", inputs, Is("strides", stride, 1), Is("pads", shape[2] / 2, kernel / 2, shape[2] / 2, kernel / 2))
            : g.Op("Conv", inputs, Is("strides", stride), Is("dilations", dilation), Is("pads", kernel / 2 * dilation, kernel / 2 * dilation));
    }
    private string Bn(string x, string prefix)
    {
        string mean = Weight(prefix + ".running_mean"), variance = Weight(prefix + ".running_var");
        string scale, bias;
        if (tensors.ContainsKey(prefix + ".weight")) { scale = Weight(prefix + ".weight"); bias = Weight(prefix + ".bias"); }
        else
        {
            int n = tensors[prefix + ".running_mean"].Values.Length;
            scale = g.Float(prefix + ".ones", [n], Enumerable.Repeat(1f, n).ToArray());
            bias = g.Float(prefix + ".zeros", [n], new float[n]);
        }
        return g.Op("BatchNormalization", [x, scale, bias, mean, variance], F("epsilon", 1e-5f));
    }
    private string Relu(string x) => g.Op("Relu", [x]);
    private string Nonlinear(string x, string prefix) => Relu(Bn(x, prefix + ".batchnorm"));
    private string Residual(string x, string prefix, int stride)
    {
        string y = Relu(Bn(Conv(x, prefix + ".conv1", stride), prefix + ".bn1"));
        y = Bn(Conv(y, prefix + ".conv2"), prefix + ".bn2");
        string skip = tensors.ContainsKey(prefix + ".shortcut.0.weight") ? Bn(Conv(x, prefix + ".shortcut.0", stride), prefix + ".shortcut.1") : x;
        return Relu(g.Op("Add", [y, skip]));
    }
    internal void Save(string path)
    {
        string x = g.Op("Unsqueeze", ["features", g.Long(1)]);
        x = Relu(Bn(Conv(x, "head.conv1"), "head.bn1"));
        for (int layer = 1; layer <= 2; layer++) for (int block = 0; block < 2; block++) x = Residual(x, $"head.layer{layer}.{block}", block == 0 ? 2 : 1);
        x = Relu(Bn(Conv(x, "head.conv2", 2), "head.bn2"));
        x = g.Reshape(x, 1, 320, -1);
        x = Conv(x, "xvector.tdnn.linear", 2);
        x = Nonlinear(x, "xvector.tdnn.nonlinear");
        int[] layers = [12, 24, 16];
        for (int b = 0; b < layers.Length; b++)
        {
            for (int l = 1; l <= layers[b]; l++)
            {
                string p = $"xvector.block{b + 1}.tdnnd{l}";
                string h = Nonlinear(x, p + ".nonlinear1");
                h = Nonlinear(Conv(h, p + ".linear1"), p + ".nonlinear2");
                string local = Conv(h, p + ".cam_layer.linear_local", dilation: b == 2 ? 2 : b + 1);
                // Pad an incomplete segment explicitly and divide by its real-frame count.
                // This also handles utterances shorter than the 100-frame pooling kernel.
                string pad = g.Long(0, 0, 0, 0, 0, 99);
                string zero = g.Op("ConstantOfShape", [g.Op("Shape", [h])]);
                string ones = g.Op("Add", [zero, g.Float(p + ".one", [1], [1])]);
                string counts = g.Op("AveragePool", [g.Op("Pad", [ones, pad])], Is("kernel_shape", 100), Is("strides", 100));
                string pooled = g.Op("AveragePool", [g.Op("Pad", [h, pad])], Is("kernel_shape", 100), Is("strides", 100));
                pooled = g.Op("Div", [pooled, counts]);
                string scales = g.Float(p + ".repeat", [3], [1, 1, 100]);
                pooled = g.Op("Resize", [pooled, "", scales], S("mode", "nearest"), S("coordinate_transformation_mode", "asymmetric"), S("nearest_mode", "floor"));
                string length = g.Op("Gather", [g.Op("Shape", [h]), g.Long(2)], I("axis", 0));
                pooled = g.Op("Slice", [pooled, g.Long(0), length, g.Long(2)]);
                string context = g.Op("Add", [g.Mean(h, 2), pooled]);
                string mask = g.Op("Sigmoid", [Conv(Relu(Conv(context, p + ".cam_layer.linear1")), p + ".cam_layer.linear2")]);
                x = g.Concat(x, g.Op("Mul", [local, mask]));
            }
            string prefix = $"xvector.transit{b + 1}";
            x = Conv(Nonlinear(x, prefix + ".nonlinear"), prefix + ".linear");
        }
        x = Nonlinear(x, "xvector.out_nonlinear");
        string meanValue = g.Mean(x, 2);
        string deviation = g.Op("Sub", [x, meanValue]);
        string varianceValue = g.Mean(g.Op("Mul", [deviation, deviation]), 2);
        // Unbiased pooling and BN → ReLU order match the pinned ModelScope black-box outputs.
        string lengthFloat = g.Op("Cast", [g.Op("Gather", [g.Op("Shape", [x]), g.Long(2)], I("axis", 0))], I("to", 1));
        string one = g.Float("one", [1], [1]);
        varianceValue = g.Op("Mul", [varianceValue, g.Op("Div", [lengthFloat, g.Op("Sub", [lengthFloat, one])])]);
        x = g.Concat(meanValue, g.Op("Sqrt", [varianceValue]));
        x = Bn(Conv(x, "xvector.dense.linear"), "xvector.dense.nonlinear.batchnorm");
        int dimension = tensors["xvector.dense.linear.weight"].Shape[0];
        x = g.Reshape(x, 1, dimension);
        g.Save(path, x, dimension);
    }
}
