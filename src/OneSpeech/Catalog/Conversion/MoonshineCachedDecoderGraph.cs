using OneSpeech.Moonshine;

namespace OneSpeech.Catalog.Conversion;

internal sealed partial class MoonshineGraphWriter
{
    // Cross-attention keys/values depend on the encoder, not the generated token.
    internal void DecoderProject(MoonshineConfiguration c)
    {
        Input("encoder", 1, 1, "frames", c.EncoderWidth);
        Input("encoder_positions", 7, "frames");
        string memory = Op("Add", ["encoder", Op("Gather", [Weight("model.decoder.pos_emb.weight"), "encoder_positions"], A("axis", 0))]);
        if (weights.Tensors.ContainsKey("model.decoder.proj.weight")) memory = Linear(memory, "model.decoder.proj", false);
        var projected = new List<string>();
        for (int layer = 0; layer < c.DecoderLayers; layer++)
        {
            string p = "model.decoder.layers." + layer + ".encoder_attn";
            projected.Add(Heads(Linear(memory, p + ".k_proj", false), c.DecoderHeads, c.DecoderHeadDimension));
            projected.Add(Heads(Linear(memory, p + ".v_proj", false), c.DecoderHeads, c.DecoderHeadDimension));
        }
        Output(Op("Concat", projected.ToArray(), A("axis", 0)), "cross", c.DecoderLayers * 2, c.DecoderHeads, "frames", c.DecoderHeadDimension);
    }

    // One query; bounded cache storage keeps execution-plan shapes stable across token positions.
    internal void DecoderStep(MoonshineConfiguration c)
    {
        Input("tokens", 7, 1, 1);
        Input("cos", 1, 1, 1, 1, c.RotaryDimension);
        Input("sin", 1, 1, 1, 1, c.RotaryDimension);
        Input("self_mask", 1, 1, 1, 1, "cache");
        Input("cross_mask", 1, 1, 1, 1, "frames");
        Input("cache_index", 7, 1, c.DecoderHeads, c.DecoderHeadDimension);
        Input("past", 1, c.DecoderLayers * 2, "cache", c.DecoderHeads, c.DecoderHeadDimension);
        Input("cross", 1, c.DecoderLayers * 2, c.DecoderHeads, "frames", c.DecoderHeadDimension);
        var updates = new List<string>();
        string x = Op("Gather", [Weight("model.decoder.embed_tokens.weight"), "tokens"], A("axis", 0));
        for (int layer = 0; layer < c.DecoderLayers; layer++)
        {
            string p = "model.decoder.layers." + layer;
            string pastK = Reshape(Slice("past", layer * 2, layer * 2 + 1, 0), -1, c.DecoderHeads, c.DecoderHeadDimension);
            string pastV = Reshape(Slice("past", layer * 2 + 1, layer * 2 + 2, 0), -1, c.DecoderHeads, c.DecoderHeadDimension);
            string crossK = Slice("cross", layer * 2, layer * 2 + 1, 0);
            string crossV = Slice("cross", layer * 2 + 1, layer * 2 + 2, 0);
            string n = Norm(x, p + ".input_layernorm");
            string q = Rotary(Heads(Linear(n, p + ".self_attn.q_proj", false), c.DecoderHeads, c.DecoderHeadDimension), c.RotaryDimension, c.DecoderHeads);
            string k = Reshape(Rotary(Heads(Linear(n, p + ".self_attn.k_proj", false), c.DecoderHeads, c.DecoderHeadDimension), c.RotaryDimension, c.DecoderHeads), 1, c.DecoderHeads, c.DecoderHeadDimension);
            string v = Reshape(Heads(Linear(n, p + ".self_attn.v_proj", false), c.DecoderHeads, c.DecoderHeadDimension), 1, c.DecoderHeads, c.DecoderHeadDimension);
            updates.Add(k); updates.Add(v);
            string keys = Reshape(Transpose(Op("ScatterElements", [pastK, "cache_index", k], A("axis", 0)), 1, 0, 2), 1, c.DecoderHeads, -1, c.DecoderHeadDimension);
            string values = Reshape(Transpose(Op("ScatterElements", [pastV, "cache_index", v], A("axis", 0)), 1, 0, 2), 1, c.DecoderHeads, -1, c.DecoderHeadDimension);
            x = Op("Add", [x, Linear(Attend(q, keys, values, c.DecoderHeads, c.Width, "self_mask").Result, p + ".self_attn.o_proj", false)]);
            q = Heads(Linear(Norm(x, p + ".post_attention_layernorm"), p + ".encoder_attn.q_proj", false), c.DecoderHeads, c.DecoderHeadDimension);
            x = Op("Add", [x, Linear(Attend(q, crossK, crossV, c.DecoderHeads, c.Width, "cross_mask").Result, p + ".encoder_attn.o_proj", false)]);
            x = Op("Add", [x, Mlp(Norm(x, p + ".final_layernorm"), p, c.Width * 4, true)]);
        }
        Output(Op("Concat", updates.ToArray(), A("axis", 0)), "new_cache", c.DecoderLayers * 2, c.DecoderHeads, c.DecoderHeadDimension);
        string projection = weights.Tensors.ContainsKey("proj_out.weight") ? "proj_out.weight" : "model.decoder.embed_tokens.weight";
        Output(Op("MatMul", [Norm(x, "model.decoder.norm"), Transpose(Weight(projection), 1, 0)]), "logits", 1, 1, 32768);
    }
}
