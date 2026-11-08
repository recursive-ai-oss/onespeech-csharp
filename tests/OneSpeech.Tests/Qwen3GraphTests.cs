using System.Buffers.Binary;
using System.Text;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Qwen3;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using Xunit;

namespace OneSpeech.Tests;

public sealed class Qwen3GraphTests
{
    [Theory]
    [InlineData(false, .7f, -.1f)]
    [InlineData(true, .69921875f, -.10009765625f)]
    public void VocabularyHeadSelectsLastPositionAndRoundsOnlyWhenRequested(bool bf16, float first, float second)
    {
        string directory = Path.Combine(Path.GetTempPath(), "qwen-head-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            WriteWeights(Path.Combine(directory, "model.safetensors"));
            using var weights = new QwenWeights(directory);
            var config = new QwenConfiguration(2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 0);
            var writer = new QwenGraphWriter(weights, bf16);
            writer.LanguageHead(config);
            string path = Path.Combine(directory, "head.onemodel");
            writer.Save(path, TestContext.Current.CancellationToken);
            using var session = Session.Load(path, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: TestContext.Current.CancellationToken);
            using var input = Tensor.FromSpan<float>([1, 2, .3f, .4f], Shape.Create(1, 2, 2));
            using var outputs = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = input });
            var logits = outputs["logits"].Data<float>();
            Assert.Equal(2, logits.Length);
            Assert.InRange(Math.Abs(logits[0] - first), 0, 1e-7f);
            Assert.InRange(Math.Abs(logits[1] - second), 0, 1e-7f);
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AudioEmbeddingsAreSplicedByTheModelGraph(bool empty)
    {
        string directory = Path.Combine(Path.GetTempPath(), "qwen-embedding-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            WriteWeights(Path.Combine(directory, "model.safetensors"));
            using var weights = new QwenWeights(directory);
            var writer = new QwenGraphWriter(weights);
            writer.Embeddings(new QwenConfiguration(2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 0));
            string path = Path.Combine(directory, "embeddings.onemodel"); writer.Save(path, TestContext.Current.CancellationToken);
            using var session = Session.Load(path, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: TestContext.Current.CancellationToken);
            using var tokens = Tensor.FromSpan<long>([0, 1, 0], Shape.Create(1, 3));
            using var audio = Tensor.FromSpan<float>(empty ? [] : [9, 8], Shape.Create(1, empty ? 0 : 1, 2));
            using var positions = Tensor.FromSpan<long>(empty ? [] : [0, 1], Shape.Create(empty ? 0 : 1, 2));
            using var output = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokens, ["audio"] = audio, ["audio_positions"] = positions });
            float[] expected = empty ? [1f, 1f, 1f, -1f, 1f, 1f] : [1f, 1f, 9f, 8f, 1f, 1f];
            Assert.Equal(expected, output["embeddings"].Data<float>().ToArray());
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("can't", "can't")]
    [InlineData("John's", "John's")]
    [InlineData("Stuff-it", "Stuffit")]
    [InlineData("word—two", "wordtwo")]
    [InlineData("A_B", "AB")]
    [InlineData("e\u0301", "e")]
    [InlineData("café", "café")]
    public void EnglishAlignmentUnitsMatchBlackBoxCleaning(string written, string spoken) => Assert.Equal(spoken, Qwen3ForcedAligner.SpokenForm(written));

    [Fact]
    public void QwenCatalogContainsBothPrecisionsAndTheTimestampHead()
    {
        foreach (string id in new[] { "Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B" })
        {
            var entry = OneSpeech.Catalog.ModelCatalog.Default.Get(id);
            Assert.Equal("Apache-2.0", entry.License); Assert.False(entry.Gated); Assert.False(entry.IsDefault);
            Assert.All(entry.ConvertedFiles, file => Assert.InRange(file.Size, 1, int.MaxValue));
            Assert.Contains(entry.ConvertedFiles, file => file.Name == "encoder.onemodel");
            Assert.Contains(entry.ConvertedFiles, file => file.Name == "embeddings.onemodel");
            Assert.Contains(entry.ConvertedFiles, file => file.Name == "head.onemodel");
            Assert.Equal(id.Contains("ForcedAligner", StringComparison.Ordinal) ? 11 : 22, entry.ConvertedFiles.Length);
            if (!id.Contains("ForcedAligner", StringComparison.Ordinal)) Assert.Contains(entry.ConvertedFiles, file => file.Name == "decoder-0-bf16.onemodel");
        }
    }

    [Fact]
    public void DuplicateShardedTensorsAreRefusedWithoutLeakingFiles()
    {
        string directory = Path.Combine(Path.GetTempPath(), "qwen-shards-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            WriteWeights(Path.Combine(directory, "model-00001-of-00002.safetensors"));
            WriteWeights(Path.Combine(directory, "model-00002-of-00002.safetensors"));
            Assert.Throws<ModelIntegrityException>(() => new QwenWeights(directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(0x3f808000u, 0x3f800000u)]
    [InlineData(0x3f818000u, 0x3f820000u)]
    [InlineData(0x80000000u, 0x80000000u)]
    [InlineData(0x7f800000u, 0x7f800000u)]
    public void PositionalRoundingIsNearestEven(uint input, uint expected) => Assert.Equal(expected, BitConverter.SingleToUInt32Bits(QwenRounding.BFloat16(BitConverter.UInt32BitsToSingle(input))));

    private static void WriteWeights(string path)
    {
        byte[] header = Encoding.UTF8.GetBytes("""{"thinker.lm_head.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"thinker.model.embed_tokens.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]}}""");
        using var file = File.Create(path);
        byte[] prefix = new byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)header.Length);
        file.Write(prefix); file.Write(header); file.Write([0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f, 0x80, 0xbf, 0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f, 0x80, 0xbf]);
    }
}
