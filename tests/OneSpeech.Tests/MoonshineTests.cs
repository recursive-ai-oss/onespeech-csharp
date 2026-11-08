using System.Text;
using System.Text.Json;
using OneSpeech.Moonshine;
using Xunit;

namespace OneSpeech.Tests;

public sealed class MoonshineTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;
    [Fact]
    public void PinsRefuseCommunityModelsAndOptionsAreBounded()
    {
        Assert.Throws<CapabilityNotSupportedException>(() => MoonshineProvenance.Get("moonshine-ai/moonshine-tiny-zh"));
        Assert.Equal(64, MoonshineProvenance.Get("moonshine-ai/moonshine-tiny").Sha256.Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => new MoonshineOptions { IntraOpThreads = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MoonshineOptions { TokensPerSecond = double.NaN }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MoonshineOptions { TokensPerSecond = 33 }.Validate());
    }
    [Fact]
    public void TokenizerRejectsHostileCountsIdsAndDepth()
    {
        Assert.ThrowsAny<JsonException>(() => new MoonshineTokenizer(Encoding.UTF8.GetBytes("{")));
        Assert.Throws<InvalidDataException>(() => new MoonshineTokenizer(new byte[8 * 1024 * 1024 + 1]));
        Assert.ThrowsAny<JsonException>(() => new MoonshineTokenizer(Encoding.UTF8.GetBytes(new string('[', 32) + new string(']', 32))));
        foreach (int id in new[] { -1, 32768, int.MaxValue })
            Assert.Throws<InvalidDataException>(() => new MoonshineTokenizer(Encoding.UTF8.GetBytes("{\"model\":{\"type\":\"BPE\",\"byte_fallback\":true,\"vocab\":{\"a\":" + id + "}}}")));
        Assert.ThrowsAny<JsonException>(() => new MoonshineTokenizer([0xFF, 0xFE]));
    }
    [Fact]
    public void TokenizerMatchesReleasedReferenceExactly()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_MOONSHINE_TINY");
        Assert.SkipWhen(string.IsNullOrWhiteSpace(directory), "Set ONESPEECH_MOONSHINE_TINY to the pinned local tokenizer directory.");
        var tokenizer = new MoonshineTokenizer(directory!);
        using var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "moonshine-tokenizer-reference.json")));
        foreach (var item in cases.RootElement.GetProperty("cases").EnumerateArray())
        {
            int[] ids = tokenizer.Encode(item.GetProperty("text").GetString()!, cancellationToken: Token);
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()), ids);
            Assert.Equal(item.GetProperty("decoded").GetString(), tokenizer.Decode(ids, false, Token));
        }
        foreach (var item in cases.RootElement.GetProperty("decodeCases").EnumerateArray())
            Assert.Equal(item.GetProperty("decoded").GetString(), tokenizer.Decode(item.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray(), cancellationToken: Token));
        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Encode(new string('a', 16385), cancellationToken: Token));
        Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode([32768], cancellationToken: Token));
        Assert.Throws<OperationCanceledException>(() => tokenizer.Encode("test", cancellationToken: new CancellationToken(true)));
    }
    [Fact]
    public void StreamingCheckpointEmitsPartialsAndImmutableFinals()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_MOONSHINE_STREAMING_TINY");
        Assert.SkipWhen(string.IsNullOrWhiteSpace(directory), "Set ONESPEECH_MOONSHINE_STREAMING_TINY to a verified converted model.");
        using var model = MoonshineRecognizer.Load(directory!, cancellationToken: Token);
        using var vad = new OneSpeech.Vad.SileroVad(cancellationToken: Token);
        using var session = model.CreateLiveSession(vad, "en");
        var clip = OneAudio.Audio.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "1089-134686-0001.flac"), new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 }, Token);
        var events = new List<LiveEvent>();
        for (int start = 0; start < clip.Samples.Length; start += 1600)
            events.AddRange(session.Push(new AudioBuffer(clip.Samples.Slice(start, Math.Min(1600, clip.Samples.Length - start)), 16000), Token));
        events.AddRange(session.Complete(Token));
        Assert.NotEmpty(events.OfType<PartialEvent>());
        Assert.NotEmpty(events.OfType<FinalEvent>());
        Assert.All(events.OfType<FinalEvent>(), f => Assert.All(f.Segment.Words, w => { Assert.Null(w.Start); Assert.Null(w.End); Assert.Equal(TimingMethod.None, w.TimingMethod); }));
        foreach (var group in events.OfType<PartialEvent>().GroupBy(p => p.UtteranceId)) Assert.Equal(Enumerable.Range(1, group.Count()).Select(i => (long)i), group.Select(p => p.Revision));
        Assert.All(events.OfType<FinalEvent>(), f => Assert.True(f.Segment.End <= f.AudioTime));
        Assert.Equal(TimeSpan.FromSeconds(clip.Samples.Length / 16000d), session.AudioTime);
    }
}
