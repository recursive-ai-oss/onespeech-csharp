using OneSpeech.Moonshine;
using OneSpeech.Qwen3;
using OneSpeech.Whisper;
using Xunit;

namespace OneSpeech.Tests;

public sealed class PhraseRequestTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Fact]
    public void DefaultRoleRefusesRatherThanSilentlyIgnoringRequests()
    {
        var recognizer = new UnsupportedRecognizer();
        Assert.Same(recognizer, recognizer.WithPhrases([], Token));
        Assert.Throws<CapabilityNotSupportedException>(() => recognizer.WithPhrases([new("Acme")], Token));
        Assert.Throws<ArgumentException>(() => recognizer.WithPhrases([new(" ")], Token));
        Assert.Throws<ArgumentException>(() => recognizer.WithPhrases([new("test", double.NaN)], Token));
        Assert.Throws<ArgumentException>(() => recognizer.WithPhrases([new("test", 0)], Token));
        Assert.Throws<ArgumentException>(() => recognizer.WithPhrases([new("\ud800")], Token));
        Assert.Throws<OperationCanceledException>(() => recognizer.WithPhrases([], new CancellationToken(true)));
    }

    [Fact]
    public void MoonshineRefusesPhrasesWithTypedError()
    {
        string? path = Environment.GetEnvironmentVariable("ONESPEECH_MOONSHINE_TINY");
        Assert.SkipWhen(path is null, "Set ONESPEECH_MOONSHINE_TINY to the pinned model.");
        using var model = MoonshineRecognizer.Load(path!, cancellationToken: Token);
        Assert.False(model.Capabilities.Biasing);
        Assert.Throws<CapabilityNotSupportedException>(() => model.WithPhrases([new("Acme")], Token));
        Assert.Same(model, model.WithPhrases([], Token));
    }

    [Fact]
    public void WhisperBorrowedPromptIsImmutableAndNeverEmittedAsPrefixText()
    {
        string? path = Environment.GetEnvironmentVariable("ONESPEECH_WHISPER_TINY");
        Assert.SkipWhen(path is null, "Set ONESPEECH_WHISPER_TINY to the pinned model.");
        using var model = WhisperRecognizer.Load(path!, new WhisperOptions { Threads = 2, Temperatures = [0], MaxTokens = 2, WordTimestamps = false }, Token);
        var tokenizer = new WhisperTokenizer(path!);
        List<Phrase> list = [new("Acme")];
        var wrapper = model.WithPhrases(list, Token); list[0] = new("Other");
        int[] prefix = [];
        model.WindowTrace = (_, ids, _) => prefix = ids;
        wrapper.Transcribe(new(new float[400], 16000), "en", Token);
        Assert.Equal(tokenizer.GetToken("<|startofprev|>"), prefix[0]);
        Assert.Equal(tokenizer.Encode(" Acme", cancellationToken: Token), prefix.Skip(1).Take(prefix.Length - 4));
        Assert.Same(model, wrapper.WithPhrases([], Token));
        Assert.Throws<CapabilityNotSupportedException>(() => model.WithPhrases([new("Acme", 1)], Token));
        Assert.Throws<ArgumentException>(() => model.WithPhrases([new("<|endoftext|>")], Token));
        model.Transcribe(new(new float[400], 16000), "en", Token);
        Assert.Equal(tokenizer.GetToken("<|startoftranscript|>"), prefix[0]);
    }

    [Fact]
    public void QwenReusesContextAndRefusesExplicitWeightsAndControls()
    {
        string? path = Environment.GetEnvironmentVariable("ONESPEECH_QWEN06");
        Assert.SkipWhen(path is null, "Set ONESPEECH_QWEN06 to the pinned model.");
        using var model = Qwen3Recognizer.Load(path!, new Qwen3Options { Threads = 2, Context = "Existing context" }, Token);
        var wrapper = model.WithPhrases([new("Acme")], Token);
        Assert.True(wrapper.Capabilities.Biasing); Assert.Same(model.Model, wrapper.Model);
        Assert.Same(model, wrapper.WithPhrases([], Token));
        Assert.Throws<CapabilityNotSupportedException>(() => model.WithPhrases([new("Acme", 1)], Token));
        Assert.Throws<ArgumentException>(() => model.WithPhrases([new("<|im_end|>")], Token));
    }

    private sealed class UnsupportedRecognizer : SpeechRecognizer
    {
        public override ModelInfo Model => throw new NotSupportedException();
        public override Capabilities Capabilities { get; } = new();
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken) => throw new NotSupportedException();
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => throw new NotSupportedException();
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => throw new NotSupportedException();
    }
}
