using System.Text;
using System.Text.Json;
using OneSpeech.FastConformer;
using OneSpeech.Moonshine;
using OneSpeech.Qwen3;
using OneSpeech.Whisper;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ModelDataReaderTests
{
    [Theory]
    [InlineData("whisper", "vocab.json", "{}")]
    [InlineData("whisper", "vocab.json", "[]")]
    [InlineData("whisper", "vocab.json", "{\"a\":true}")]
    [InlineData("whisper", "added_tokens.json", "{\"\":256}")]
    [InlineData("whisper", "added_tokens.json", "[]")]
    [InlineData("whisper", "merges.txt", "a b\n")]
    [InlineData("qwen", "vocab.json", "[]")]
    [InlineData("qwen", "vocab.json", "{\"a\":true}")]
    [InlineData("qwen", "tokenizer_config.json", "{}")]
    [InlineData("qwen", "tokenizer_config.json", "{\"tokenizer_class\":false}")]
    public void MalformedModelDocumentsHaveTypedFailures(string family, string name, string text)
    {
        string directory = Create();
        try
        {
            File.WriteAllText(Path.Combine(directory, name), text);
            Assert.Throws<InvalidDataException>(() => Load(family, directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("whisper")]
    [InlineData("qwen")]
    public void CompleteByteVocabularyStillRoundTrips(string family)
    {
        string directory = Create();
        try
        {
            const string text = " a 🙂中文\r\n";
            if (family == "whisper") { var tokenizer = new WhisperTokenizer(directory); Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text, cancellationToken: TestContext.Current.CancellationToken))); }
            else { var tokenizer = new Qwen3Tokenizer(directory); Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text, cancellationToken: TestContext.Current.CancellationToken))); }
            File.WriteAllBytes(Path.Combine(directory, "merges.txt"), [0xff]);
            Assert.Throws<InvalidDataException>(() => Load(family, directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("{\"d_model\":true}")]
    public void ConfigurationShapesHaveTypedFailures(string input)
    {
        string directory = Create();
        try
        {
            File.WriteAllText(Path.Combine(directory, "config.json"), input);
            Assert.Throws<InvalidDataException>(() => WhisperConfiguration.Load(directory));
            Assert.Throws<InvalidDataException>(() => QwenConfiguration.Load(directory));
            Assert.Throws<ModelIntegrityException>(() => NemotronAsrRecognizer.ReadPrompts(Path.Combine(directory, "config.json")));
            File.WriteAllBytes(Path.Combine(directory, "upstream.txt"), [0xff]);
            Assert.Throws<ModelIntegrityException>(() => MoonshineProvenance.Verify(directory, false, TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData("\"no_timestamps_token_id\":128", "\"no_timestamps_token_id\":255")]
    [InlineData("\"eos_token_id\":1", "\"eos_token_id\":256")]
    [InlineData("\"<|en|>\"", "\"x\"")]
    [InlineData("[[0,0]]", "[[0]]")]
    [InlineData("[[0,0]]", "[[1,0]]")]
    [InlineData("[-1,0]", "[256]")]
    public void GenerationMetadataRejectsInvalidIdsAndHeads(string before, string after)
    {
        string directory = Create();
        const string valid = """
            {"decoder_start_token_id":2,"eos_token_id":1,"task_to_id":{"transcribe":3,"translate":4},"no_timestamps_token_id":128,"lang_to_id":{"<|en|>":5},"suppress_tokens":[-1,0],"alignment_heads":[[0,0]]}
            """;
        var configuration = new WhisperConfiguration(4, 1, 1, 2, 2, 80, 256, 8, 8);
        try
        {
            string path = Path.Combine(directory, "generation_config.json");
            File.WriteAllText(path, valid);
            Assert.Equal(2, WhisperGenerationConfiguration.Load(directory, configuration).Start);
            File.WriteAllText(path, valid.Replace(before, after, StringComparison.Ordinal));
            Assert.Throws<InvalidDataException>(() => WhisperGenerationConfiguration.Load(directory, configuration));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void UnusedOrdinaryEmptyTokenIsNotAnEmptySpecialToken()
    {
        string directory = Create();
        try
        {
            string path = Path.Combine(directory, "vocab.json");
            string original = File.ReadAllText(path);
            File.WriteAllText(path, original[..^1] + ",\"\":256}");
            Assert.Empty(new WhisperTokenizer(directory).Decode([256]));
            File.WriteAllText(Path.Combine(directory, "added_tokens.json"), "{\"\":256}");
            Assert.Throws<InvalidDataException>(() => new WhisperTokenizer(directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    private static void Load(string family, string directory)
    {
        if (family == "whisper") _ = new WhisperTokenizer(directory);
        else _ = new Qwen3Tokenizer(directory);
    }
    private static string Create()
    {
        string directory = Directory.CreateTempSubdirectory("onespeech-model-data-test-").FullName;
        using (var file = File.Create(Path.Combine(directory, "vocab.json")))
        using (var json = new Utf8JsonWriter(file))
        {
            // Deterministic synthetic byte vocabulary; no speech or publisher fixture.
            json.WriteStartObject(); int extra = 256;
            for (int b = 0; b < 256; b++)
            {
                char character = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)b : (char)extra++;
                json.WriteNumber(character.ToString(), b);
            }
            json.WriteEndObject();
        }
        File.WriteAllText(Path.Combine(directory, "merges.txt"), "#version: 0.2\n", new UTF8Encoding(false, true));
        File.WriteAllText(Path.Combine(directory, "added_tokens.json"), "{}");
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), "{\"tokenizer_class\":\"Qwen2Tokenizer\",\"added_tokens_decoder\":{}}");
        return directory;
    }
}
