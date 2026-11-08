using System.Buffers.Binary;
using OneSpeech.Catalog;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Whisper;
using Xunit;

namespace OneSpeech.Tests;

public sealed class Int8Tests
{
    [Fact]
    public void PerChannelWeightsAreTransposedWithNearestEvenRounding()
    {
        float[] source = [127, .5f, 1.5f, 254, -1, -3];
        var result = Int8Weights.Transpose(source, 2, 3);
        Assert.Equal(new byte[] { 127, 127, 0, 0, 2, 254 }, result.Values);
        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(result.Scales));
        Assert.Equal(2, BinaryPrimitives.ReadSingleLittleEndian(result.Scales.AsSpan(4)));
        Assert.Equal(source, new float[] { 127, .5f, 1.5f, 254, -1, -3 });
        var repeated = Int8Weights.Transpose(source, 2, 3);
        Assert.Equal(result.Values, repeated.Values);
        Assert.Equal(result.Scales, repeated.Scales);
    }

    [Fact]
    public void ZeroAndSubnormalChannelsHavePositiveFiniteScales()
    {
        var zero = Int8Weights.Transpose(new float[4], 2, 2);
        Assert.Equal(new byte[4], zero.Values);
        Assert.Equal(1, BinaryPrimitives.ReadSingleLittleEndian(zero.Scales));
        var tiny = Int8Weights.Transpose([float.Epsilon, -float.Epsilon], 1, 2);
        Assert.True(BinaryPrimitives.ReadSingleLittleEndian(tiny.Scales) > 0);
        Assert.Equal(new byte[] { 1, 255 }, tiny.Values);
        Assert.Throws<ModelIntegrityException>(() => Int8Weights.Transpose([float.NaN], 1, 1));
        Assert.Throws<ModelIntegrityException>(() => Int8Weights.Transpose([float.PositiveInfinity], 1, 1));
        Assert.Throws<ModelIntegrityException>(() => Int8Weights.Transpose([], 0, 1));
        Assert.Throws<ModelIntegrityException>(() => Int8Weights.Transpose([1], 1, 2));
    }

    [Theory]
    [InlineData("openai/whisper-small")]
    [InlineData("openai/whisper-large-v3-turbo")]
    [InlineData("moonshine-ai/moonshine-streaming-tiny")]
    public void CatalogVariantsReusePublisherPinsButHaveDistinctRuntimeDigests(string id)
    {
        var catalog = ModelCatalog.Default;
        var original = catalog.Get(id);
        var quantized = catalog.Get(id + "-int8");
        Assert.False(quantized.IsDefault);
        Assert.Equal(original.Source, quantized.Source);
        Assert.Equal(original.Revision, quantized.Revision);
        Assert.Equal(original.Files, quantized.Files);
        Assert.Equal(original.License, quantized.License);
        Assert.Equal(original.Attribution, quantized.Attribution);
        Assert.Equal(ModelTier.Tier1, quantized.Tier);
        Assert.Equal(original.ConvertedFiles.Length - (id.EndsWith("turbo", StringComparison.Ordinal) ? 1 : 0), quantized.ConvertedFiles.Length);
        foreach (var file in quantized.ConvertedFiles)
            Assert.DoesNotContain(original.ConvertedFiles, f => f.Sha256 == file.Sha256);
        Assert.NotEqual(quantized.Id, catalog.GetDefault(original.Family).Id);
    }

    [Fact]
    public void WhisperProvenancePreservesVariantIdAndPublisherSource()
    {
        var original = ModelCatalog.Default.Get("openai/whisper-small");
        var model = WhisperCheckpoint.Provenance([original.Id + "-int8", original.Revision, original.Files.Single(f => f.Name == "model.safetensors").Sha256]);
        Assert.Equal(original.Id + "-int8", model.Id);
        Assert.Equal(original.Source, model.Source);
        Assert.Throws<ModelIntegrityException>(() => WhisperCheckpoint.Provenance([original.Id + "-int8", new string('0', 40), model.Files[0].Sha256]));
    }
}
