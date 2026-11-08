using static SpeakerGraph;

// Omnilingual ASR §4.1–4.2: wav2vec 2.0 feature encoder, Transformer, linear CTC head.
internal sealed class OmniAsrConverter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private readonly HashSet<string> added = new(StringComparer.Ordinal);
    private string W(string name)
    {
        if (added.Add(name)) { var t = tensors[name]; g.Float(name, t.Shape, t.Values); }
        return name;
    }
    private string Scalar(string name, float value) => g.Float(name, [], [value]);
    private string Transpose(string x, params long[] axes) => g.Op("Transpose", [x], Is("perm", axes));
    private string Norm(string x, string prefix) => g.Op("LayerNormalization", [x, W(prefix + ".weight"), W(prefix + ".bias")], I("axis", -1), F("epsilon", 1e-5f));
    private string Linear(string x, string prefix) => g.Op("Add", [g.Op("MatMul", [x, Transpose(W(prefix + ".weight"), 1, 0)]), W(prefix + ".bias")]);
    private string Gelu(string x) => g.Op("Mul", [g.Op("Mul", [x, "half"]), g.Op("Add", ["one", g.Op("Erf", [g.Op("Mul", [x, "inverseSqrtTwo"])])])]);
    private string Heads(string x) => Transpose(g.Reshape(x, 0, -1, 16, 64), 0, 2, 1, 3);

    internal void Save(string path)
    {
        Scalar("half", 0.5f); Scalar("one", 1); Scalar("inverseSqrtTwo", 1 / MathF.Sqrt(2)); Scalar("attentionScale", 0.125f);
        // Baevski et al. 2020 §2: utterance-level zero mean/unit variance before temporal convolutions.
        string mean = g.Mean("waveform", 2);
        string centered = g.Op("Sub", ["waveform", mean]);
        string variance = g.Mean(g.Op("Mul", [centered, centered]), 2);
        string x = g.Op("Div", [centered, g.Op("Sqrt", [g.Op("Add", [variance, Scalar("waveformEpsilon", 1e-5f)])])]);
        for (int i = 0; i < 7; i++)
        {
            string prefix = "encoder_frontend.feature_extractor.layers." + i;
            x = g.Op("Conv", [x, W(prefix + ".conv.weight"), W(prefix + ".conv.bias")], Is("strides", i == 0 ? 5 : 2));
            x = Transpose(Gelu(Norm(Transpose(x, 0, 2, 1), prefix + ".layer_norm")), 0, 2, 1);
        }
        x = Linear(Norm(Transpose(x, 0, 2, 1), "encoder_frontend.post_extract_layer_norm"), "encoder_frontend.model_dim_proj");
        const string position = "encoder_frontend.pos_encoder.conv";
        var v = tensors[position + ".weight_v"];
        var magnitude = tensors[position + ".weight_g"].Values;
        if (!v.Shape.SequenceEqual([1024, 64, 128]) || magnitude.Length != 128) throw new InvalidDataException("Unexpected positional convolution shape.");
        double[] squares = new double[128];
        for (int i = 0; i < v.Values.Length; i++) squares[i % 128] += (double)v.Values[i] * v.Values[i];
        float[] weights = new float[v.Values.Length];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(v.Values[i] * magnitude[i % 128] / Math.Sqrt(squares[i % 128]));
        string convolution = g.Op("Conv", [Transpose(x, 0, 2, 1), g.Float("positionWeight", v.Shape, weights), W(position + ".bias")], I("group", 16), Is("pads", 64, 64));
        // Even kernel gives one extra output; remove the rightmost frame to preserve sequence length.
        convolution = g.Op("Slice", [convolution, g.Long(0), g.Long(-1), g.Long(2)]);
        x = g.Op("Add", [x, Gelu(Transpose(convolution, 0, 2, 1))]);
        for (int i = 0; i < 24; i++)
        {
            string prefix = "encoder.layers." + i;
            string n = Norm(x, prefix + ".self_attn_layer_norm");
            string q = g.Op("Mul", [Heads(Linear(n, prefix + ".self_attn.q_proj")), "attentionScale"]);
            string k = Heads(Linear(n, prefix + ".self_attn.k_proj"));
            string vHead = Heads(Linear(n, prefix + ".self_attn.v_proj"));
            string probabilities = g.Op("Softmax", [g.Op("MatMul", [q, Transpose(k, 0, 1, 3, 2)])], I("axis", -1));
            string attended = g.Reshape(Transpose(g.Op("MatMul", [probabilities, vHead]), 0, 2, 1, 3), 0, -1, 1024);
            x = g.Op("Add", [x, Linear(attended, prefix + ".self_attn.output_proj")]);
            n = Norm(x, prefix + ".ffn_layer_norm");
            x = g.Op("Add", [x, Linear(Gelu(Linear(n, prefix + ".ffn.inner_proj")), prefix + ".ffn.output_proj")]);
        }
        x = Linear(Norm(x, "encoder.layer_norm"), "final_proj");
        g.Save(path, x, 9812, inputName: "waveform", inputShape: [1, 1, "samples"], outputName: "logits", outputShape: [1, "frames", 9812]);
    }
}
