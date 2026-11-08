using OneSpeech.Formats;

namespace OneSpeech.Catalog.Conversion;

// Names/shapes are taken only from the pinned publisher tensor header.
internal sealed class NemotronAsrTensorSource(SafeTensorFile file) : IFloatTensorSource
{
    internal static string PublisherName(string name)
    {
        foreach (var (canonical, publisher) in new (string, string)[]
        {
            ("encoder.subsampling.0", "encoder.subsampling.conv_in"),
            ("encoder.subsampling.1.depthwise", "encoder.subsampling.layers.0.depthwise_conv"),
            ("encoder.subsampling.1.pointwise", "encoder.subsampling.layers.0.pointwise_conv"),
            ("encoder.subsampling.2.depthwise", "encoder.subsampling.layers.1.depthwise_conv"),
            ("encoder.subsampling.2.pointwise", "encoder.subsampling.layers.1.pointwise_conv")
        })
            if (name.StartsWith(canonical + ".", StringComparison.Ordinal)) return publisher + name[canonical.Length..];
        if (name.Contains(".conv.layer_norm.", StringComparison.Ordinal)) return name.Replace(".conv.layer_norm.", ".conv.norm.", StringComparison.Ordinal);
        if (name.StartsWith("prompt_projector.", StringComparison.Ordinal)) return name;
        return ParakeetTensorSource.PublisherName(name);
    }
    public long[] GetShape(string name) => file.Tensors[PublisherName(name)].Shape.ToArray();
    public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken) => file.CopyFloat32To(PublisherName(name), destination, cancellationToken);
}

internal static class NemotronAsrConverter
{
    internal static readonly FastConformerDimensions Dimensions = new(Vocabulary: 13088, DurationCount: 0);
    internal static void Convert(string input, string output, CancellationToken cancellationToken, bool streaming = false)
    {
        Directory.CreateDirectory(output);
        using var file = new SafeTensorFile(File.OpenRead(Path.Combine(input, "model.safetensors")));
        var source = new NemotronAsrTensorSource(file);
        if (streaming)
        {
            new NemotronAsrGraphBuilder(source, Dimensions).EncoderStep(Path.Combine(output, "encoder-step0.onemodel"), 0, 12, true, cancellationToken);
            new NemotronAsrGraphBuilder(source, Dimensions).EncoderStep(Path.Combine(output, "encoder-step1.onemodel"), 12, 12, false, cancellationToken);
        }
        else
        {
            new NemotronAsrGraphBuilder(source, Dimensions).Encoder(Path.Combine(output, "encoder-part0.onemodel"), 0, 12, true, cancellationToken);
            new NemotronAsrGraphBuilder(source, Dimensions).Encoder(Path.Combine(output, "encoder-part1.onemodel"), 12, 12, false, cancellationToken);
        }
        new NemotronAsrGraphBuilder(source, Dimensions).PromptProjector(Path.Combine(output, "prompt.onemodel"), cancellationToken);
        new FastConformerGraphBuilder(source, Dimensions).Predictor(Path.Combine(output, "predictor.onemodel"), cancellationToken);
        new FastConformerGraphBuilder(source, Dimensions).Joint(Path.Combine(output, "joint.onemodel"), cancellationToken);
        using var embedding = File.Create(Path.Combine(output, "embedding.f32"));
        file.CopyFloat32To("decoder.embedding.weight", embedding, cancellationToken);
    }
}
