using OneSpeech.Qwen3;

namespace OneSpeech.Catalog.Conversion;

internal static class QwenConverter
{
    internal static void Convert(string directory, string output, CancellationToken cancellationToken, Qwen3Precision precision = Qwen3Precision.Binary32)
    {
        Directory.CreateDirectory(output);
        string Name(string name) => Path.Combine(output, name + (precision == Qwen3Precision.BFloat16 ? "-bf16" : "") + ".onemodel");
        var config = QwenConfiguration.Load(directory);
        using var weights = new QwenWeights(directory);
        var front = new QwenGraphWriter(weights, precision == Qwen3Precision.BFloat16);
        front.EncoderFront(config); front.Save(Name("encoder-front"), cancellationToken);
        var encoder = new QwenGraphWriter(weights, precision == Qwen3Precision.BFloat16);
        encoder.Encoder(config); encoder.Save(Name("encoder"), cancellationToken);
        var embedding = new QwenGraphWriter(weights, precision == Qwen3Precision.BFloat16);
        embedding.Embeddings(config); embedding.Save(Name("embeddings"), cancellationToken);
        var head = new QwenGraphWriter(weights, precision == Qwen3Precision.BFloat16);
        head.LanguageHead(config); head.Save(Name("head"), cancellationToken);
        for (int begin = 0; begin < config.Layers; begin += 4)
        {
            var decoder = new QwenGraphWriter(weights, precision == Qwen3Precision.BFloat16);
            decoder.Decoder(config, begin, Math.Min(config.Layers, begin + 4));
            decoder.Save(Name("decoder-" + begin / 4), cancellationToken);
        }
    }
}
