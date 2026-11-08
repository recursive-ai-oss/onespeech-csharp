using System.Text.Json;
using OneSpeech.Whisper;
using Xunit;

namespace OneSpeech.Tests;

public sealed class WhisperModelTests
{
    private static string ModelDirectory()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_WHISPER_TINY");
        Assert.SkipWhen(string.IsNullOrWhiteSpace(directory), "Set ONESPEECH_WHISPER_TINY to a verified, converted tiny model directory to run model tests.");
        return directory!;
    }

    [Fact]
    public void TokenizerMatchesPinnedReferenceCasesAndRoundTrips()
    {
        var tokenizer = new WhisperTokenizer(ModelDirectory());
        using var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures/whisper-tokenizer-reference.json")));
        foreach (var item in cases.RootElement.EnumerateArray())
        {
            string text = item.GetProperty("text").GetString()!;
            int[] actual = tokenizer.Encode(text, allowSpecialTokens: true, TestContext.Current.CancellationToken);
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()), actual);
            Assert.Equal(item.GetProperty("decoded").GetString(), tokenizer.Decode(actual, includeSpecialTokens: true));
            Assert.Equal(text, tokenizer.Decode(actual, includeSpecialTokens: true));
        }
        Assert.Throws<ArgumentException>(() => tokenizer.Encode("<|startoftranscript|>", cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void EmptyTimestampSlicesDoNotConditionTheNextWindow()
    {
        string directory = ModelDirectory();
        var tokenizer = new WhisperTokenizer(directory);
        using var recognizer = WhisperRecognizer.Load(directory, cancellationToken: TestContext.Current.CancellationToken);
        int timestamp = tokenizer.GetToken("<|0.00|>");
        var empty = recognizer.Slice([timestamp], 0, 480000, "en");
        Assert.Empty(empty.Segments);
        Assert.Empty(empty.ContextTokens);
        Assert.Equal(480000, empty.Advance);
        int[] text = tokenizer.Encode(" Hello", cancellationToken: TestContext.Current.CancellationToken);
        int[] spaces = tokenizer.Encode(" ", cancellationToken: TestContext.Current.CancellationToken);
        Assert.Empty(recognizer.Slice([timestamp, .. spaces, timestamp + 50], 0, 480000, "en").ContextTokens);
        Assert.Empty(recognizer.Slice([timestamp + 50, .. text, timestamp + 50, timestamp + 51], 0, 480000, "en").ContextTokens);
        int[] accepted = [timestamp, .. text, timestamp + 50];
        var combined = recognizer.Slice([.. accepted, timestamp + 50], 0, 480000, "en");
        Assert.Single(combined.Segments);
        Assert.Equal(accepted, combined.ContextTokens);
        Assert.Equal(text, combined.TextTokens);
    }

    [Fact]
    public void TinyTranscribesTheAttributedShortClip()
    {
        string directory = ModelDirectory();
        var clip = OneAudio.Audio.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures/1089-134686-0001.flac"), new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 }, TestContext.Current.CancellationToken);
        using var recognizer = WhisperRecognizer.Load(directory, cancellationToken: TestContext.Current.CancellationToken);
        var transcript = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", TestContext.Current.CancellationToken);
        Assert.Equal(" Stuffed into you, his belly, cowtled him.", transcript.Text);
        Assert.NotEmpty(transcript.Words);
        Assert.All(transcript.Words, word => { Assert.Equal(TimingMethod.Dtw, word.TimingMethod); Assert.NotNull(word.Start); Assert.NotNull(word.End); Assert.True(word.Start <= word.End); Assert.InRange(word.Confidence!.Value, 0, 1); });
        Assert.Equal("en", transcript.Language);
        Assert.Throws<CapabilityNotSupportedException>(() => recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "not-a-language", TestContext.Current.CancellationToken));
    }
}
