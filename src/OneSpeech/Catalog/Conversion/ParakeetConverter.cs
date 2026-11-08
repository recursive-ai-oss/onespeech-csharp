using OneSpeech.Formats;

namespace OneSpeech.Catalog.Conversion;

// Publisher tensor names are data from the pinned safetensors header.
internal sealed class ParakeetTensorSource(SafeTensorFile file) : IFloatTensorSource
{
    internal static string PublisherName(string name)
    {
        if (name.StartsWith("encoder.layers.", StringComparison.Ordinal))
        {
            string[] parts = name.Split('.');
            string prefix = "encoder.layers." + parts[2] + ".";
            string suffix = string.Join('.', parts.Skip(3));
            foreach (var (canonical, publisher) in new (string, string)[]
            {
                ("ff1.norm", "norm_feed_forward1"), ("ff2.norm", "norm_feed_forward2"),
                ("ff1.in", "feed_forward1.linear1"), ("ff1.out", "feed_forward1.linear2"),
                ("ff2.in", "feed_forward2.linear1"), ("ff2.out", "feed_forward2.linear2"),
                ("attention_norm", "norm_self_att"), ("conv_norm", "norm_conv"), ("final_norm", "norm_out"),
                ("attention.q", "self_attn.q_proj"), ("attention.k", "self_attn.k_proj"),
                ("attention.v", "self_attn.v_proj"), ("attention.position", "self_attn.relative_k_proj"),
                ("attention.out", "self_attn.o_proj"), ("attention.u", "self_attn.bias_u"), ("attention.v_bias", "self_attn.bias_v"),
                ("conv.pointwise_in", "conv.pointwise_conv1"), ("conv.pointwise_out", "conv.pointwise_conv2"),
                ("conv.depthwise", "conv.depthwise_conv"), ("conv.batch_norm.mean", "conv.norm.running_mean"),
                ("conv.batch_norm.variance", "conv.norm.running_var"), ("conv.batch_norm", "conv.norm")
            })
                if (suffix == canonical || suffix.StartsWith(canonical + ".", StringComparison.Ordinal)) return prefix + publisher + suffix[canonical.Length..];
        }
        foreach (var (canonical, publisher) in new (string, string)[]
        {
            ("encoder.subsampling.0", "encoder.subsampling.layers.0"),
            ("encoder.subsampling.1.depthwise", "encoder.subsampling.layers.2"),
            ("encoder.subsampling.1.pointwise", "encoder.subsampling.layers.3"),
            ("encoder.subsampling.2.depthwise", "encoder.subsampling.layers.5"),
            ("encoder.subsampling.2.pointwise", "encoder.subsampling.layers.6"),
            ("encoder.subsampling.projection", "encoder.subsampling.linear"),
            ("joint.encoder", "encoder_projector"), ("joint.predictor", "decoder.decoder_projector"), ("joint.output", "joint.head")
        })
            if (name.StartsWith(canonical + ".", StringComparison.Ordinal)) return publisher + name[canonical.Length..];
        if (name.StartsWith("predictor.layers.", StringComparison.Ordinal))
        {
            string[] parts = name.Split('.');
            string projection = parts[3] == "input" ? "ih" : "hh";
            return "decoder.lstm." + parts[4] + "_" + projection + "_l" + parts[2];
        }
        throw new InvalidDataException("Unmapped Parakeet tensor: " + name);
    }

    public long[] GetShape(string name) => file.Tensors[PublisherName(name)].Shape.ToArray();
    public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken) => file.CopyFloat32To(PublisherName(name), destination, cancellationToken);
}

internal static class ParakeetConverter
{
    internal static void Convert(string input, string output, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(output);
        using var file = new SafeTensorFile(File.OpenRead(Path.Combine(input, "model.safetensors")));
        var source = new ParakeetTensorSource(file);
        var dimensions = new FastConformerDimensions();
        new FastConformerGraphBuilder(source, dimensions).Encoder(Path.Combine(output, "encoder-part0.onemodel"), 0, 12, true, cancellationToken);
        new FastConformerGraphBuilder(source, dimensions).Encoder(Path.Combine(output, "encoder-part1.onemodel"), 12, 12, false, cancellationToken);
        new FastConformerGraphBuilder(source, dimensions).Predictor(Path.Combine(output, "predictor.onemodel"), cancellationToken);
        new FastConformerGraphBuilder(source, dimensions).Joint(Path.Combine(output, "joint.onemodel"), cancellationToken);
        using var embedding = File.Create(Path.Combine(output, "embedding.f32"));
        file.CopyFloat32To("decoder.embedding.weight", embedding, cancellationToken);
    }
}
