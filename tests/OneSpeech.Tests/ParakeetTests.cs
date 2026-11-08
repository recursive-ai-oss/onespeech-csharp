using System.Text;
using System.Text.Json;
using OneSpeech.Catalog;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Decoding;
using OneSpeech.FastConformer;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ParakeetTests
{
    [Theory]
    [InlineData("encoder.layers.12.attention.position.weight", "encoder.layers.12.self_attn.relative_k_proj.weight")]
    [InlineData("encoder.layers.0.conv.batch_norm.variance", "encoder.layers.0.conv.norm.running_var")]
    [InlineData("encoder.subsampling.2.pointwise.weight", "encoder.subsampling.layers.6.weight")]
    [InlineData("predictor.layers.1.hidden.bias", "decoder.lstm.bias_hh_l1")]
    [InlineData("joint.predictor.weight", "decoder.decoder_projector.weight")]
    public void PublisherTensorMappingIsExplicit(string canonical, string publisher) => Assert.Equal(publisher, ParakeetTensorSource.PublisherName(canonical));

    [Fact]
    public void VocabularyRetainsTokenWordRangesAndPunctuation()
    {
        var vocabulary = new ParakeetVocabulary(Vocabulary());
        FrameToken[] tokens = [new(0, 1, 2, .8f), new(1, 2, 3, .9f), new(2, 4, 5, .9f), new(3, 4, 5, .7f)];
        Assert.Equal("Hello, world", vocabulary.Decode([0, 1, 2, 3]));
        Assert.Equal(" ⁇ ", vocabulary.Decode([8191]));
        Assert.Equal(" ⁇  Hel ⁇ ", vocabulary.Decode([8191, 0, 8191]));
        var words = vocabulary.Words(tokens);
        Assert.Equal([new WordTokenRange("Hello,", 0, 3), new WordTokenRange("world", 3, 1)], words);
        var timed = NativeWordTiming.Map(vocabulary.TimingTokens(tokens), vocabulary.TimingRanges(tokens, words), TimeSpan.FromMilliseconds(80), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TimeSpan.FromMilliseconds(80), timed[0].Start);
        Assert.Equal(TimeSpan.FromMilliseconds(240), timed[0].End);
        Assert.Equal(TimingMethod.Native, timed[0].TimingMethod);
        var model = new ModelInfo { Id = "fixture", Source = "https://example.org/model", Revision = new string('a', 40), Tier = ModelTier.Tier2, License = "CC-BY-4.0", Attribution = "Synthetic fixture" };
        var transcript = ParakeetRecognizer.CreateTranscript(new AudioBuffer(new float[16000], 16000), [.. tokens], vocabulary, model, "en", TestContext.Current.CancellationToken);
        Assert.Equal(2, transcript.Words.Length);
        Assert.Equal(transcript.Words, transcript.Segments.Single().Words);
        Assert.All(transcript.Words, word => Assert.Equal(TimingMethod.Native, word.TimingMethod));
        Assert.Equal("Hello, world", transcript.Text);
        Assert.Empty(vocabulary.Words([]));
        Assert.Throws<InvalidDataException>(() => vocabulary.Words([new(8192, 0, 1, 1)]));
    }

    [Fact]
    public void HostileVocabularyIsBoundedAndRejected()
    {
        Assert.Throws<InvalidDataException>(() => new ParakeetVocabulary(new byte[2 * 1024 * 1024 + 1]));
        foreach (byte[] bytes in new[] { "{}"u8.ToArray(), "[]"u8.ToArray(), new byte[] { 0xff }, Vocabulary()[..100], Encoding.UTF8.GetBytes("{\"model\":{\"type\":\"BPE\",\"byte_fallback\":false,\"vocab\":{\"a\":8192}}}") })
            Assert.Throws<InvalidDataException>(() => new ParakeetVocabulary(bytes));
        string original = Encoding.UTF8.GetString(Vocabulary());
        Assert.Throws<InvalidDataException>(() => new ParakeetVocabulary(Encoding.UTF8.GetBytes(original.Replace("\"lo\":1", "\"lo\":0", StringComparison.Ordinal))));
    }

    [Fact]
    public void Tier2ConvertedMetadataIsPublisherOnlyAndCannotBecomeDefault()
    {
        var entry = ModelCatalog.Default.Get("nvidia/parakeet-tdt-0.6b-v3");
        Assert.Equal(ModelTier.Tier2, entry.Tier); Assert.False(entry.IsDefault); Assert.False(entry.Gated);
        Assert.Equal("CC-BY-4.0", entry.License);
        Assert.Contains("NVIDIA", entry.Attribution, StringComparison.Ordinal);
        Assert.Contains(entry.Source, entry.Attribution, StringComparison.Ordinal);
        Assert.Contains("CC-BY-4.0", entry.Attribution, StringComparison.Ordinal);
        Assert.Throws<ModelNotFoundException>(() => ModelCatalog.Default.GetDefault("parakeet-tdt"));
        Assert.All(entry.Files.Concat(entry.ConvertedFiles), file => { Assert.Null(file.MirrorUrl); Assert.StartsWith(entry.Source + "/resolve/" + entry.Revision + "/", file.UpstreamUrl, StringComparison.Ordinal); });
        Assert.All(entry.ConvertedFiles, file => Assert.Contains(entry.Files, input => input.UpstreamUrl == file.UpstreamUrl));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entry with { IsDefault = true }]));
        var output = entry.ConvertedFiles[0];
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entry with { ConvertedFiles = [output with { MirrorUrl = "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/test/encoder.onemodel" }] }]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entry with { ConvertedFiles = [output with { UpstreamUrl = "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/test/encoder.onemodel" }] }]));
    }

    internal static byte[] Vocabulary()
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject(); writer.WriteStartObject("model"); writer.WriteString("type", "BPE"); writer.WriteBoolean("byte_fallback", false); writer.WriteStartObject("vocab");
            string[] first = ["▁Hel", "lo", ",", "▁world"];
            for (int i = 0; i < 8192; i++) writer.WriteNumber(i < first.Length ? first[i] : i == 8191 ? "<unk>" : "piece" + i, i);
            writer.WriteEndObject(); writer.WriteEndObject(); writer.WriteEndObject();
        }
        return stream.ToArray();
    }
}
