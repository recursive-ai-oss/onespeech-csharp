using OneSpeech.Moonshine;

namespace OneSpeech.Catalog.Conversion;

internal sealed partial class MoonshineGraphWriter
{
    private string Slice(string x, long begin, long end, long axis) => Op("Slice", [x, Ints(begin), Ints(end), Ints(axis)]);
    private string Silu(string x) => Op("Mul", [x, Op("Sigmoid", [x])]);
    // Released Transformers 5.22.2 attention outputs match adjacent-pair rotation, not split-half rotation.
    private string Rotary(string x, int dimension, int heads)
    {
        string rotated = Slice(x, 0, dimension, 3);
        string paired = Reshape(rotated, 1, heads, -1, dimension / 2, 2);
        string half = Reshape(Op("Concat", [Op("Neg", [Slice(paired, 1, 2, 4)]), Slice(paired, 0, 1, 4)], A("axis", 4)), 1, heads, -1, dimension);
        rotated = Op("Add", [Op("Mul", [rotated, "cos"]), Op("Mul", [half, "sin"])]);
        return Op("Concat", [rotated, Slice(x, dimension, long.MaxValue, 3)], A("axis", 3));
    }
    private string MeanVariance(string x, long[] axes)
    {
        string centered = Op("Sub", [x, Op("ReduceMean", [x], As("axes", axes), A("keepdims", 1))]);
        string variance = Op("ReduceMean", [Op("Mul", [centered, centered])], As("axes", axes), A("keepdims", 1));
        return Op("Div", [centered, Op("Sqrt", [Op("Add", [variance, Scalar(1e-5f)])])]);
    }
    private string Mlp(string x, string prefix, int intermediate, bool gated)
    {
        string y = Linear(x, prefix + ".mlp.fc1");
        // The published tensor has value channels first and SiLU gate channels second, confirmed by first-logit parity.
        y = gated ? Op("Mul", [Slice(y, 0, intermediate, 2), Silu(Slice(y, intermediate, intermediate * 2, 2))]) : Gelu(y);
        return Linear(y, prefix + ".mlp.fc2");
    }
    private string Attention(string x, string memory, string prefix, int heads, int width, bool rotary, int rotaryDimension, string? mask = null)
    {
        string q = Heads(Linear(x, prefix + ".q_proj", false), heads, width / heads);
        string k = Heads(Linear(memory, prefix + ".k_proj", false), heads, width / heads);
        string v = Heads(Linear(memory, prefix + ".v_proj", false), heads, width / heads);
        if (rotary) { q = Rotary(q, rotaryDimension, heads); k = Rotary(k, rotaryDimension, heads); }
        var attended = Attend(q, k, v, heads, width, mask);
        if (diagnostics && prefix == "model.encoder.layers.0.self_attn") Output(attended.Probabilities, "attention-0", 1, heads, "frames", "frames");
        return Linear(attended.Result, prefix + ".o_proj", false);
    }
    internal void Encoder(MoonshineConfiguration config)
    {
        if (config.Streaming) { StreamingEncoder(config); return; }
        Input("audio", 1, 1, 1, "samples");
        Input("cos", 1, 1, 1, "frames", config.RotaryDimension);
        Input("sin", 1, 1, 1, "frames", config.RotaryDimension);
        string x = Op("Tanh", [Op("Conv", ["audio", Weight("model.encoder.conv1.weight")], As("strides", 64))]);
        x = Op("Add", [Op("Mul", [MeanVariance(x, [1, 2]), Reshape(Weight("model.encoder.groupnorm.weight"), 1, -1, 1)]), Reshape(Weight("model.encoder.groupnorm.bias"), 1, -1, 1)]);
        x = Gelu(Op("Conv", [x, Weight("model.encoder.conv2.weight"), Weight("model.encoder.conv2.bias")], As("strides", 3)));
        x = Transpose(Gelu(Op("Conv", [x, Weight("model.encoder.conv3.weight"), Weight("model.encoder.conv3.bias")], As("strides", 2))), 0, 2, 1);
        if (diagnostics) Output(x, "stage_0", 1, "frames", config.Width);
        for (int layer = 0; layer < config.EncoderLayers; layer++)
        {
            string prefix = "model.encoder.layers." + layer;
            string n = Norm(x, prefix + ".input_layernorm");
            x = Op("Add", [x, Attention(n, n, prefix + ".self_attn", config.EncoderHeads, config.Width, true, config.RotaryDimension)]);
            x = Op("Add", [x, Mlp(Norm(x, prefix + ".post_attention_layernorm"), prefix, config.Width * 4, false)]);
            if (diagnostics) Output(x, "stage_" + (layer + 1), 1, "frames", config.Width);
        }
        Output(Norm(x, "model.encoder.layer_norm"), "encoder", 1, "frames", config.Width);
    }
    internal void Decoder(MoonshineConfiguration config)
    {
        Input("tokens", 7, 1, "tokens");
        Input("encoder", 1, 1, "frames", config.EncoderWidth);
        string memory = "encoder";
        if (config.Streaming)
        {
            Input("encoder_positions", 7, "frames");
            memory = Op("Add", [memory, Op("Gather", [Weight("model.decoder.pos_emb.weight"), "encoder_positions"], A("axis", 0))]);
            if (weights.Tensors.ContainsKey("model.decoder.proj.weight")) memory = Linear(memory, "model.decoder.proj", false);
        }
        Input("cos", 1, 1, 1, "tokens", config.RotaryDimension);
        Input("sin", 1, 1, 1, "tokens", config.RotaryDimension);
        Input("mask", 1, 1, 1, "tokens", "tokens");
        string x = Op("Gather", [Weight("model.decoder.embed_tokens.weight"), "tokens"], A("axis", 0));
        if (diagnostics) Output(x, "stage_0", 1, "tokens", config.Width);
        for (int layer = 0; layer < config.DecoderLayers; layer++)
        {
            string prefix = "model.decoder.layers." + layer;
            string n = Norm(x, prefix + ".input_layernorm");
            x = Op("Add", [x, Attention(n, n, prefix + ".self_attn", config.DecoderHeads, config.Width, true, config.RotaryDimension, "mask")]);
            x = Op("Add", [x, Attention(Norm(x, prefix + ".post_attention_layernorm"), memory, prefix + ".encoder_attn", config.DecoderHeads, config.Width, false, 0)]);
            x = Op("Add", [x, Mlp(Norm(x, prefix + ".final_layernorm"), prefix, config.Width * 4, true)]);
            if (diagnostics) Output(x, "stage_" + (layer + 1), 1, "tokens", config.Width);
        }
        x = Norm(x, "model.decoder.norm");
        string projection = weights.Tensors.ContainsKey("proj_out.weight") ? "proj_out.weight" : "model.decoder.embed_tokens.weight";
        Output(Op("MatMul", [x, Transpose(Weight(projection), 1, 0)]), "logits", 1, "tokens", 32768);
    }
}
