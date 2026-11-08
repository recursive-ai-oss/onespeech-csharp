using System.Text.Json;
using System.Text.Json.Serialization;
using OneSpeech.Catalog;
using OneSpeech.Live;
using OneSpeech.Pipeline;
using OneSpeech.Vad;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ErgonomicsTests
{
    [Theory]
    [InlineData("en-US", "en")]
    [InlineData("EN", "en")]
    [InlineData("english", "en")]
    [InlineData("eng", "en")]
    [InlineData("zh-Hant-TW", "zh")]
    [InlineData("Chinese", "zh")]
    [InlineData("eng_Latn", "en")]
    public void LanguageNormalizationIsShared(string input, string expected) => Assert.Equal(expected, LanguageTag.Normalize(input));

    [Theory]
    [InlineData("zz")]
    [InlineData("")]
    [InlineData("en--US")]
    public void UnsupportedLanguageIsTypedBeforeFileOrModelIo(string input)
    {
        Assert.Throws<CapabilityNotSupportedException>(() => LanguageTag.Normalize(input));
        Assert.Throws<CapabilityNotSupportedException>(() => Speech.Align("missing", "test", input, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => Speech.Transcribe("missing", input, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => Speech.Identify("missing", input, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void EveryJsonRouteHasOneShapeAndValueEquality()
    {
        var word = new Word { Text = "x", Start = TimeSpan.FromTicks(123), End = TimeSpan.FromTicks(456), TimingMethod = TimingMethod.Native };
        var document = new Transcript { Text = "x", Words = [word], Segments = [new Segment { Text = "x", Words = [word] }],
            Turns = [new Turn { Text = "x", Speaker = SpeakerLabel.Unassigned, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Words = [word] }],
            Models = [Info("silero/vad-v6.2.3")] };
        string json = JsonSerializer.Serialize(document, ApiJsonContext.Default.Transcript);
        Assert.Equal(document.ToJson(), json);
        var copy = JsonSerializer.Deserialize(json, ApiJsonContext.Default.Transcript);
        Assert.Equal(document, copy);
        Assert.Equal(document.GetHashCode(), copy!.GetHashCode());
        Assert.Equal(document, Transcript.FromJson(json));
        Assert.NotEqual(document, document with { Text = "different" });
        var exception = Assert.Throws<InvalidTranscriptJsonException>(() => (document with { Words = [word with { End = TimeSpan.Zero }] }).ToJson());
        Assert.Contains("words[0].start/end", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OmittedWireArraysUseDocumentedDefaultsWhileNullArraysRefuse()
    {
        var minimum = Transcript.FromJson("""{"schemaVersion":1,"transcript":{"text":"x","segments":[{"text":"x"}],"turns":[{"speaker":{"id":"s"},"text":"x","start":0,"end":1}],"models":[{"id":"m","source":"s","revision":"r","tier":"Tier1","license":"MIT","attribution":"a"}]}}""");
        Assert.Empty(minimum.Words); Assert.Empty(minimum.Segments[0].Words); Assert.Empty(minimum.Turns[0].Words); Assert.Empty(minimum.Models[0].Files);
        Assert.Equal(minimum, Transcript.FromJson(minimum.ToJson()));
        foreach (string field in new[] { "words", "segments", "turns", "models" })
            Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("{\"schemaVersion\":1,\"transcript\":{\"text\":\"x\",\"" + field + "\":null}}"));
        var error = Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("""{"schemaVersion":1,"transcript":{"text":"a","words":[{"text":"a","start":2,"end":1}]}}"""));
        Assert.Contains("words[0].start/end", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RolesComposeAndSessionsBorrowThem()
    {
        using SpeechRecognizer recognizer = new EmptyRecognizer();
        using VoiceActivityDetector vad = new EmptyVad();
        using var session = recognizer.CreateLiveSession(vad, "en-US");
        Assert.Empty(session.Push(new AudioBuffer(new float[512], 16000), TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => session.Push(new AudioBuffer(new float[441], 44100), TestContext.Current.CancellationToken));
        session.Dispose();
        Assert.Empty(vad.Detect(new AudioBuffer(ReadOnlyMemory<float>.Empty, 16000), TestContext.Current.CancellationToken));
        using var silero = new SileroVad(cancellationToken: TestContext.Current.CancellationToken);
        Assert.Throws<ArgumentException>(() => silero.Detect(new AudioBuffer(new float[441], 44100), TestContext.Current.CancellationToken));
        _ = ModelCatalog.Default.Get(silero.Model.Id);
    }

    [Fact]
    public async Task LoaderFamilyCheckAndPrefetchCancellationNeedNoWeights()
    {
        var wrong = new ResolvedModel("missing", Info("openai/whisper-tiny"));
        Assert.Throws<CapabilityNotSupportedException>(() => Embedders.CampPlusEmbedder.Load(wrong, cancellationToken: TestContext.Current.CancellationToken));
        using var stop = new CancellationTokenSource(); stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await Speech.PrefetchAsync(model: "openai/whisper-tiny", cancellationToken: stop.Token));
        var unknown = Assert.Throws<ModelNotFoundException>(() => ModelCatalog.Default.Get("whisper-tiny"));
        Assert.Equal("whisper-tiny", unknown.ModelId);
        Assert.Contains("Did you mean", unknown.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EncodedSpoolClosesItsWriterBeforeReplayAndLeavesInputOpen()
    {
        using var input = new MemoryStream([1, 2, 3]);
        string path;
        await using (var spool = await SpeechServices.SpoolAsync(input, TestContext.Current.CancellationToken))
        {
            path = spool.Name;
            using var replay = File.OpenRead(path);
            Assert.Equal(1, replay.ReadByte());
            Assert.True(input.CanRead);
        }
        Assert.False(File.Exists(path));
    }

    [Fact]
    public void PrefetchPlanCoversConditionalHappyPathCompanions()
    {
        var store = new ModelStore();
        var fixedEnglish = SpeechServices.PrefetchModels(store, "en-US", "openai/whisper-tiny");
        Assert.Equal(["openai/whisper-tiny", "Qwen/Qwen3-ForcedAligner-0.6B", "facebook/omniASR-CTC-300M", "iic/speech_campplus_sv_en_voxceleb_16k"], fixedEnglish);
        var auto = SpeechServices.PrefetchModels(store, null, "moonshine-ai/moonshine-streaming-tiny");
        Assert.Contains("iic/speech_campplus_sv_en_voxceleb_16k", auto);
        Assert.Contains("iic/speech_campplus_sv_zh-cn_16k-common", auto);
        Assert.Equal(["Qwen/Qwen3-ForcedAligner-0.6B"], SpeechServices.PrefetchModels(store, "EN", "Qwen/Qwen3-ForcedAligner-0.6B"));
    }

    private static ModelInfo Info(string id)
    {
        var e = ModelCatalog.Default.Get(id);
        return new ModelInfo { Id = e.Id, Source = e.Source, Revision = e.Revision, License = e.License, Tier = e.Tier, Attribution = e.Attribution,
            Files = [.. e.Files.Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 })] };
    }

    private sealed class EmptyVad : VoiceActivityDetector
    {
        public override ModelInfo Model => Info("silero/vad-v6.2.3");
        public override System.Collections.Immutable.ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken = default) => [];
    }
    private sealed class EmptyRecognizer : SpeechRecognizer
    {
        public override ModelInfo Model => Info("openai/whisper-tiny");
        public override Capabilities Capabilities => new() { Languages = ["en"] };
        public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default) => new() { Text = "" };
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default) => [];
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default) => "en";
    }
}

[JsonSerializable(typeof(Transcript))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class ApiJsonContext : JsonSerializerContext;
