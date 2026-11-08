using OneSpeech.Moonshine;

namespace OneSpeech.Catalog.Conversion;

internal sealed partial class MoonshineGraphWriter
{
    private string GammaNorm(string x, string prefix) => Op("LayerNormalization", [x, Op("Add", [Weight(prefix + ".gamma"), Scalar(1)])], A("axis", -1), new("epsilon", 1, Float: 1e-5f));
    private string StreamingFrontend(MoonshineConfiguration config)
    {
        // Moonshine Streaming paper §3.1: 80-sample frame CMVN, asinh, linear/SiLU, two causal stride-2 convolutions.
        string framed = Reshape("audio", 1, -1, 80);
        string centered = Op("Sub", [framed, Op("ReduceMean", [framed], As("axes", 2), A("keepdims", 1))]);
        string variance = Op("ReduceMean", [Op("Mul", [centered, centered])], As("axes", 2), A("keepdims", 1));
        if (frontendNorm >= 3) variance = Op("Mul", [variance, Scalar(80f / 79)]);
        string denominator = (frontendNorm % 3) switch
        {
            1 => Op("Add", [Op("Sqrt", [variance]), Scalar(frontendEpsilon)]),
            2 => Op("Max", [Op("Sqrt", [variance]), Scalar(frontendEpsilon)]),
            _ => Op("Sqrt", [Op("Add", [variance, Scalar(frontendEpsilon)])])
        };
        string x = Op("Div", [centered, denominator]);
        string k = Op("Exp", [Weight("model.encoder.embedder.comp.log_k")]);
        x = compressionMode switch
        {
            0 => Op("Asinh", [Op("Mul", [x, k])]),
            1 => Op("Div", [Op("Asinh", [Op("Mul", [x, k])]), k]),
            2 => Op("Asinh", [Op("Div", [x, k])]),
            3 => Op("Asinh", [x]),
            _ => x
        };
        x = Transpose(Silu(Linear(x, "model.encoder.embedder.linear", false)), 0, 2, 1);
        x = Op("Conv", [x, Weight("model.encoder.embedder.conv1.weight"), Weight("model.encoder.embedder.conv1.bias")], As("pads", 4, 0), As("strides", 2));
        if (frontendActivation % 3 == 1) x = Silu(x);
        if (frontendActivation % 3 == 2) x = Gelu(x);
        x = Op("Conv", [x, Weight("model.encoder.embedder.conv2.weight"), Weight("model.encoder.embedder.conv2.bias")], As("pads", 4, 0), As("strides", 2));
        if (frontendActivation / 3 == 1) x = Silu(x);
        if (frontendActivation / 3 == 2) x = Gelu(x);
        return Transpose(x, 0, 2, 1);
    }
    internal void FrontendProbe(MoonshineConfiguration config)
    {
        Input("audio", 1, 1, 1, "samples");
        Output(StreamingFrontend(config), "frontend", 1, "frames", config.EncoderWidth);
    }
    internal void StreamingEncoder(MoonshineConfiguration config)
    {
        Input("audio", 1, 1, 1, "samples");
        Input("local_mask", 1, 1, 1, "frames", "frames");
        Input("causal_mask", 1, 1, 1, "frames", "frames");
        string x = StreamingFrontend(config);
        if (diagnostics) Output(x, "stage_0", 1, "frames", config.EncoderWidth);
        for (int layer = 0; layer < config.EncoderLayers; layer++)
        {
            string prefix = "model.encoder.layers." + layer;
            string n = GammaNorm(x, prefix + ".input_layernorm");
            string mask = layer < 2 || layer >= config.EncoderLayers - 2 ? "local_mask" : "causal_mask";
            x = Op("Add", [x, Attention(n, n, prefix + ".self_attn", config.EncoderHeads, config.EncoderHeads * config.EncoderHeadDimension, false, 0, mask)]);
            x = Op("Add", [x, Mlp(GammaNorm(x, prefix + ".post_attention_layernorm"), prefix, config.EncoderWidth * 4, false)]);
            if (diagnostics) Output(x, "stage_" + (layer + 1), 1, "frames", config.EncoderWidth);
        }
        Output(GammaNorm(x, "model.encoder.final_norm"), "encoder", 1, "frames", config.EncoderWidth);
    }
}
