using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using OneSpeech.Qwen3;
using Xunit;

namespace OneSpeech.Tests;

public sealed class Qwen3TokenizerTests
{
    [Fact]
    public void ByteVocabularyRoundTripsNfcAndRequiresExplicitSpecialTokens()
    {
        string directory = CreateByteVocabulary();
        try
        {
            var tokenizer = new Qwen3Tokenizer(directory);
            const string text = " e\u0301 中文🙂𐐀 \r\n\t0123";
            int[] ids = tokenizer.Encode(text, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(text.Normalize(NormalizationForm.FormC), tokenizer.Decode(ids));
            Assert.Equal(256, tokenizer.GetToken("<|im_end|>"));
            Assert.True(tokenizer.IsSpecial(256));
            Assert.False(tokenizer.IsSpecial(255));
            Assert.Throws<ArgumentException>(() => tokenizer.Encode("<|im_end|>", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Equal([256], tokenizer.Encode("<|im_end|>", true, TestContext.Current.CancellationToken));
            Assert.Equal("", tokenizer.Decode(new[] { 256 }));
            Assert.Equal("<|im_end|>", tokenizer.Decode(new[] { 256 }, true));
            Assert.Equal("�", tokenizer.Decode(new[] { 0xc3 }));
            Assert.Empty(tokenizer.Encode("", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Throws<ArgumentException>(() => tokenizer.Encode("\ud800", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Throws<ArgumentException>(() => tokenizer.Encode(new string('a', 1048577), cancellationToken: TestContext.Current.CancellationToken));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode(new[] { -1 }));
            Assert.Throws<OperationCanceledException>(() => tokenizer.Encode("", cancellationToken: new CancellationToken(true)));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void HostileVocabularyAndMergeDataAreRejected()
    {
        string directory = CreateByteVocabulary();
        try
        {
            File.WriteAllText(Path.Combine(directory, "merges.txt"), "a b\na b\n");
            Assert.Throws<InvalidDataException>(() => new Qwen3Tokenizer(directory));
            File.WriteAllText(Path.Combine(directory, "merges.txt"), "");
            File.WriteAllText(Path.Combine(directory, "vocab.json"), "{\"a\":200000}");
            Assert.Throws<InvalidDataException>(() => new Qwen3Tokenizer(directory));
            File.WriteAllText(Path.Combine(directory, "vocab.json"), "{\"a\":0,\"b\":0}");
            Assert.Throws<InvalidDataException>(() => new Qwen3Tokenizer(directory));
            File.WriteAllText(Path.Combine(directory, "vocab.json"), "{\"a\":0");
            Assert.ThrowsAny<JsonException>(() => new Qwen3Tokenizer(directory));
            File.WriteAllBytes(Path.Combine(directory, "vocab.json"), [0x7b, 0x22, 0xff, 0x22, 0x3a, 0x30, 0x7d]);
            Assert.Throws<InvalidDataException>(() => new Qwen3Tokenizer(directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void PublishedTokenizerMatchesBlackBoxIdsAndDecodedText()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_TOKENIZERS");
        Assert.SkipWhen(string.IsNullOrEmpty(directory), "Set ONESPEECH_QWEN_TOKENIZERS to the pinned oracle data directory.");
        string fixturePath = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_TOKENIZER_FIXTURE") ??
            Path.Combine(AppContext.BaseDirectory, "Fixtures/qwen3-tokenizers.json");
        using var fixture = JsonDocument.Parse(File.ReadAllText(fixturePath));
        foreach (var family in fixture.RootElement.GetProperty("tokenizers").EnumerateObject())
        {
            string modelDirectory = Path.Combine(directory!, family.Name);
            foreach (string name in new[] { "vocab.json", "merges.txt", "tokenizer_config.json", "preprocessor_config.json" })
            {
                string actualHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(modelDirectory, name))));
                Assert.Equal(fixture.RootElement.GetProperty("documents").GetProperty(family.Name + "/" + name).GetString(), actualHash);
            }
            var tokenizer = new Qwen3Tokenizer(modelDirectory);
            foreach (var row in family.Value.EnumerateArray())
            {
                string text = row.GetProperty("text").GetString()!;
                int[] actual = tokenizer.Encode(text, true, TestContext.Current.CancellationToken);
                Assert.Equal(row.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()), actual);
                Assert.Equal(row.GetProperty("decoded").GetString(), tokenizer.Decode(actual, true));
                Assert.Equal(row.GetProperty("decoded_without_special").GetString(), tokenizer.Decode(actual));
            }
        }
    }

    private static string CreateByteVocabulary()
    {
        string directory = Path.Combine(Path.GetTempPath(), "qwen-tokenizer-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        using (var file = File.Create(Path.Combine(directory, "vocab.json")))
        using (var writer = new Utf8JsonWriter(file))
        {
            writer.WriteStartObject();
            int extra = 256;
            for (int b = 0; b < 256; b++)
            {
                char character = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)b : (char)extra++;
                writer.WriteNumber(character.ToString(), b);
            }
            writer.WriteEndObject();
        }
        File.WriteAllText(Path.Combine(directory, "merges.txt"), "#version: 0.2\n");
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), """
            {"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{"256":{"content":"<|im_end|>","special":true,"lstrip":false,"rstrip":false,"single_word":false,"normalized":false}}}
            """);
        return directory;
    }
}
