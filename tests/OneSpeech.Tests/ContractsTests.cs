using System.Text;
using System.Text.Json;
using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ContractsTests
{
    [Fact]
    public void RoundTripPreservesAllFieldsAndNulls()
    {
        var word = new Word { Text = " e\u0301 ", Start = TimeSpan.FromSeconds(.25), End = TimeSpan.FromSeconds(.5), Confidence = .9, TimingMethod = TimingMethod.CtcAligner, Speaker = SpeakerLabel.Unassigned, Overlap = true, Language = "fr" };
        var untimed = new Word { Text = "未知" };
        var transcript = new Transcript
        {
            Text = " e\u0301 未知", Duration = TimeSpan.FromSeconds(2), Language = "fr", Words = [word, untimed],
            Segments = [new Segment { Text = "segment", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Language = "fr", Words = [word] }],
            Turns = [new Turn { Speaker = new SpeakerLabel { Id = "s1", Name = "Speaker" }, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Text = "turn", Words = [word] }],
            Models = [new ModelInfo { Id = "model", Source = "https://example.org/model", Revision = "pinned", Files = [new ModelFileInfo { Name = "weights", Sha256 = new string('a', 64) }], Tier = ModelTier.Tier2, License = "CC-BY-4.0", Attribution = "Exact attribution\n", Gated = true }]
        };
        string json = transcript.ToJson();
        var copy = Transcript.FromJson(json);
        Assert.Equal(json, copy.ToJson());
        Assert.Equal(transcript.Text, copy.ToText());
        Assert.Equal(word, copy.Words[0]);
        Assert.Null(copy.Words[1].Start);
        Assert.Null(copy.Words[1].End);
        Assert.Null(copy.Words[1].Confidence);
        Assert.Null(copy.Words[1].Speaker);
        Assert.Contains("\"start\":null", json, StringComparison.Ordinal);
        Assert.Contains("\"start\":0.25", json, StringComparison.Ordinal);
        Assert.Equal(json, Transcript.FromJson(Encoding.UTF8.GetBytes(json)).ToJson());
    }

    [Theory]
    [InlineData("")]
    [InlineData("{")]
    [InlineData("null")]
    [InlineData("{}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\\ud800\"}}") ]
    [InlineData("{\"schemaVersion\":2,\"transcript\":{\"text\":\"\"}}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":null}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":{\"text\":null}}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"words\":[null]}}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"words\":[{\"text\":\"a\",\"start\":1}]}}")]
    [InlineData("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"duration\":1e300}}")]
    public void InvalidInputIsTyped(string json) => Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson(json));

    [Fact]
    public void ReaderBoundsBeforeMaterialization()
    {
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson(new string(' ', 8 * 1024 * 1024 + 1)));
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("{\"schemaVersion\":1,\"transcript\":{\"text\":\"" + new string('x', 65537) + "\"}}"));
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson(new string('[', 33) + new string(']', 33)));
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson(new byte[] { 34, 255, 34 }));
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("\"\ud800\""));
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("[" + string.Join(',', Enumerable.Repeat("0", 500001)) + "]"));
    }

    [Fact]
    public void CatalogUsesGeneratedMetadata()
    {
        string json = """
        {"id":"sample","family":"whisper","source":"https://example.org/model","revision":"pin","files":[{"name":"weights","size":4,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","upstreamUrl":"https://example.org/weights","mirrorUrl":null}],"tier":"Tier1","license":"MIT","attribution":"credit","gated":false,"capabilities":{"languages":["en"],"translation":true,"nativeTimestamps":true,"streaming":false,"biasing":true,"maxSpeakers":2}}
        """;
        var entry = CatalogEntry.FromJson(Encoding.UTF8.GetBytes(json));
        Assert.Equal("sample", entry.Id);
        Assert.Equal(4, entry.Files[0].Size);
        Assert.True(entry.Capabilities.Biasing);
        Assert.Equal(2, entry.Capabilities.MaxSpeakers);
        Assert.ThrowsAny<JsonException>(() => CatalogEntry.FromJson("{"u8));
    }

    [Fact]
    public void HappyPathRefusesUnknownModelsAndHonorsCancellation()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "1089-134686-0001.flac");
        Assert.Throws<ModelNotFoundException>(() => Speech.Transcribe(path, model: "missing/model"));
        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => Speech.Transcribe(path, canceled.Token));
    }

    [Fact]
    public void AudioAndErrorsHaveExplicitBehavior()
    {
        var audio = new AudioBuffer(new float[32000], 16000, 2);
        Assert.Equal(TimeSpan.FromSeconds(1), audio.Duration);
        Assert.Throws<ArgumentException>(() => new AudioBuffer(new float[3], 16000, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AudioBuffer(default, 0));
        Assert.Throws<FileNotFoundException>(() => Speech.Transcribe("missing.wav"));
        Assert.IsAssignableFrom<OneSpeechException>(new ModelNotFoundException("fetch model"));
        Assert.IsAssignableFrom<OneSpeechException>(new ModelIntegrityException("hash mismatch"));
        Assert.IsAssignableFrom<OneSpeechException>(new ModelTokenRequiredException("supply token"));
        Assert.IsAssignableFrom<OneSpeechException>(new CapabilityNotSupportedException("unsupported language"));
        Assert.IsAssignableFrom<OneSpeechException>(new UnsupportedAudioException("corrupt audio"));
    }
}
